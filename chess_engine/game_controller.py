# -*- coding: utf-8 -*-
"""对弈状态机 —— GameController + MoveRecord + 入口"""

import os
import sys
import time
import traceback
from enum import Enum
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import cchess
from ultralytics import YOLO

from chess_engine.config import Config
from chess_engine.protocol_client import ProtocolClient
from chess_engine.engine_controller import EngineController
from vision.chess_vision import VisionController
from vision.video_stream import VideoStreamThread


class MoveRecord:
    """单步走法记录"""
    __slots__ = ('uci', 'src_xy', 'dst_xy', 'is_capture', 'captured_piece', 'waste_xy')

    def __init__(self, uci: str, src_xy: Tuple[int, int] = None,
                 dst_xy: Tuple[int, int] = None,
                 is_capture: bool = False,
                 captured_piece: Optional[str] = None,
                 waste_xy: Tuple[int, int] = None):
        self.uci = uci
        self.src_xy = src_xy
        self.dst_xy = dst_xy
        self.is_capture = is_capture
        self.captured_piece = captured_piece
        self.waste_xy = waste_xy


class GameController:
    """对弈状态机 —— 消除人类先行/引擎先行两套重复代码"""

    class State(Enum):
        INIT = "init"
        READY = "ready"
        WAIT_START = "wait_start"
        HUMAN_TURN = "human_turn"
        ENGINE_TURN = "engine_turn"
        GAME_OVER = "game_over"
        UNDO = "undo"
        AUTO_SETUP = "auto_setup"

    # cchess 棋子符号 → 棋子类型映射
    _CChess_SYMBOL_TO_TYPE = {
        'K': 'Kin', 'R': 'Car', 'N': 'Hor', 'C': 'Can',
        'B': 'Ele', 'A': 'Shi', 'P': 'Paw',
        'k': 'Kin', 'r': 'Car', 'n': 'Hor', 'c': 'Can',
        'b': 'Ele', 'a': 'Shi', 'p': 'Paw',
    }

    # ------------------------------------------------------------------
    def __init__(self, proto: ProtocolClient, engine: EngineController,
                 vision: VisionController):
        self._p = proto
        self._engine = engine
        self._vis = vision

        self._state = self.State.INIT
        self._cchess: Optional[cchess.Board] = None
        self._situation: str = ""
        self._initial_fen: str = ""
        self._engine_fen: str = Config.STANDARD_OPENING_FEN  # Pikafish用
        self._cchess_fen: str = Config.STANDARD_OPENING_FEN  # cchess用 (同标准FEN)
        self._human_is_red: bool = True
        self._level: int = 1
        self._move_history: List[MoveRecord] = []
        self._use_chess_count: int = 16
        self._last_board: Optional[List[List[str]]] = None

    # ==================================================================
    # 主循环
    # ==================================================================
    def run(self):
        while True:
            if self._p.shutdown_requested():
                print("[系统] 收到退出指令，正在清理...")
                return
            if self._state == self.State.INIT:
                self._handle_init()
            elif self._state == self.State.READY:
                self._handle_ready()
            elif self._state == self.State.WAIT_START:
                self._handle_wait_start()
            elif self._state == self.State.HUMAN_TURN:
                self._handle_human_turn()
            elif self._state == self.State.ENGINE_TURN:
                self._handle_engine_turn()
            elif self._state == self.State.GAME_OVER:
                self._handle_game_over()
            elif self._state == self.State.UNDO:
                self._handle_undo()
            elif self._state == self.State.AUTO_SETUP:
                self._handle_auto_setup()

    # ==================================================================
    # 状态处理器
    # ==================================================================
    def _handle_init(self):
        print("[状态] INIT → READY")
        self._state = self.State.READY

    def _handle_ready(self):
        self._p.send("准备完毕")
        self._state = self.State.WAIT_START

    def _handle_wait_start(self):
        msg = self._p.wait_for_any_prefix("开始对弈", "残局对弈", "一键摆盘")
        if msg is None:
            return

        if msg.startswith("一键摆盘"):
            if "红" in msg:
                self._setup_fen = Config.RED_STANDARD_OPENING_FEN
            else:
                self._setup_fen = Config.STANDARD_OPENING_FEN
            self._state = self.State.AUTO_SETUP
            return

        is_endgame = msg.startswith("残局对弈")

        if is_endgame:
            # 残局对弈：提取残局名称和难度
            # 格式示例: "残局对弈1:大鹏展翅"
            params = msg.replace("残局对弈", "").split(":", 1)
            try:
                self._level = int(params[0])
            except (ValueError, IndexError):
                self._level = 1
            endgame_name = params[1] if len(params) > 1 else ""
        else:
            try:
                self._level = int(msg.replace("开始对弈", ""))
            except ValueError:
                self._level = 1
            endgame_name = ""

        self._level = max(1, min(4, self._level))
        self._engine.set_level(self._level)
        tag = f"残局「{endgame_name}」" if is_endgame else "开局"
        print(f"[系统] {tag}，难度={self._level}")

        self._reset_game_state()

        # 确定预期 FEN
        if is_endgame:
            expected_fen = Config.ENDGAME_FEN_DB.get(endgame_name)
            self._initial_fen = expected_fen  # 可能为 None
        else:
            expected_fen = Config.STANDARD_OPENING_FEN
            self._initial_fen = expected_fen

        # 识别 + 校验循环，直到通过为止
        board = None
        while True:
            if self._p.shutdown_requested():
                return
            board = self._vis.capture_stable_board()
            if board is None:
                print("初始棋盘识别失败，重试...")
                time.sleep(0.5)
                continue

            recognized_fen = self._vis.construct_fen(board, side_to_move="w")

            if is_endgame:
                if expected_fen is None:
                    # 残局库中没有，使用识别结果
                    self._initial_fen = recognized_fen
                    break
                if self._validate_against_expected(board, expected_fen):
                    break
                print("[校验] 残局校验未通过，重新识别...")
            else:
                # 根据棋盘上人类执方选择对应 FEN 校验
                human_side = self._vis.detect_side(board)
                if human_side == "red":
                    expected_fen = Config.RED_STANDARD_OPENING_FEN
                else:
                    expected_fen = Config.STANDARD_OPENING_FEN
                self._initial_fen = expected_fen
                if self._validate_against_expected(board, expected_fen):
                    break
                print("[校验] 开局校验未通过，重新识别...")

        human_color = self._vis.detect_side(board)
        self._human_is_red = (human_color == "red")
        self._vis.set_human_is_red(self._human_is_red)
        self._last_board = board

        valid, err = Config.validate_fen(self._initial_fen)
        if not valid:
            print(f"[严重] 最终 FEN 格式异常: {err}")
            self._state = self.State.GAME_OVER
            return
        print(f"[初始FEN] {self._initial_fen}")

        # 用 FEN 初始化 cchess
        try:
            self._cchess = cchess.Board(self._cchess_fen)
            print("[校验] cchess 已从引擎 FEN 初始化")
        except Exception as e:
            print(f"[警告] cchess 无法解析 FEN: {e}")
            self._initial_fen = ""
            self._cchess = cchess.Board()

        side_name = "红" if self._human_is_red else "黑"
        print(f"[判断] 人类执{side_name}方")

        self._state = (self.State.HUMAN_TURN if self._human_is_red
                       else self.State.ENGINE_TURN)

    # ------------------------------------------------------------------
    def _handle_human_turn(self):
        self._p.send("请玩家落子")
        self._p.send("文字:请落子")

        # 等待玩家落子，期间可响应悔棋
        msg = self._p.wait_for_any("玩家落子完成", "悔棋", "替我思考")
        if msg is None:
            return
        if msg == "悔棋":
            print("[悔棋] 收到悔棋请求")
            self._state = self.State.UNDO
            return

        if msg == "替我思考":
            self._p.send("文字:正在替您思考...")
            current_frame = self._vis._stream.get_latest_warped()

            try:
                human_move = self._engine.get_best_move(
                    self._situation.strip(), self._engine_fen,
                    movetime_ms=Config.THINK_FOR_HUMAN_MOVETIME)
            except RuntimeError as e:
                print(f"[替思] 引擎错误: {e}，尝试恢复...")
                if self._engine.restart(self._situation.strip(), self._engine_fen):
                    human_move = self._engine.get_best_move(
                        self._situation.strip(), self._engine_fen,
                        movetime_ms=Config.THINK_FOR_HUMAN_MOVETIME)
                else:
                    print("[替思] 引擎恢复失败")
                    self._state = self.State.GAME_OVER
                    return

            if not human_move:
                print("[替思] 引擎返回空走法")
                self._state = self.State.GAME_OVER
                return

            if not self._is_move_legal(human_move):
                print(f"[替思] 引擎走法 {human_move} 不合法")
                self._state = self.State.GAME_OVER
                return

            is_capture, captured_symbol = self._check_capture_before_push(
                self._cchess, human_move)

            self._push_engine_move(human_move)
            self._situation += f" {human_move}"
            chinese = self._cchess.move_to_notation(cchess.Move.from_uci(human_move))
            print(f"[替思] {chinese} ({human_move})")

            if self._cchess.is_check():
                print("[替思] 将军！")
                self._p.send("文字:将军")

            waste_xy = None
            captured_label = None
            if is_capture:
                dst_uci = human_move[2:]
                _, dst_xy = self._vis.detect_piece_at(dst_uci, current_frame)
                # 替人类吃子 → 放到人类废棋区（左侧），悔棋时也是从人类废棋区找回
                waste_xy = self._get_human_waste_xy()
                captured_label = self._symbol_to_label(captured_symbol)
                self._p.send(f"棋子移动:{dst_xy},{waste_xy}")
                print(f"[替思] 吃子！({captured_symbol}) 放入人类废棋区")

            src_uci = human_move[:2]
            dst_uci = human_move[2:]
            _, src_xy = self._vis.detect_piece_at(src_uci, current_frame)
            dst_pixel = self._vis.uci_to_pixel_public(dst_uci)
            self._p.send(f"棋子移动:{src_xy},{dst_pixel}")
            self._p.wait_for("运动完成")

            self._move_history.append(MoveRecord(
                human_move, src_xy, dst_pixel,
                is_capture=is_capture, captured_piece=captured_label,
                waste_xy=waste_xy))

            self._fen_before_human = self._cchess.fen()

            if self._check_game_over():
                return

            self._last_board = self._vis.apply_uci_move(self._last_board, human_move)
            self._state = self.State.ENGINE_TURN
            return

        self._p.send("文字:小桌思考中...")

        human_move = None
        while human_move is None:
            board_now = None
            while board_now is None:
                if self._p.shutdown_requested():
                    return
                board_now = self._vis.capture_stable_board()
                if board_now is None:
                    print("棋盘识别失败，重试...")
                    time.sleep(0.5)

            human_move = self._vis.get_move(self._last_board, board_now)

            if human_move is None:
                print("[警告] 未检测到走法变化，重新识别...")
                continue

            if not self._vis.move_looks_valid(human_move):
                print(f"[警告] 未识别到有效走法: {human_move!r}")
                human_move = self._handle_move_recognition_failure(human_move)
                continue

            if not self._is_move_legal(human_move):
                print(f"[警告] 人类非法走法: {human_move}")
                self._vis.print_board(self._last_board)
                self._vis.print_board(board_now)
                human_move = self._handle_illegal_move(human_move)
                continue

            # 后验校验：从 cchess 权威状态推算预期棋子数，与识别结果比对
            is_capture, captured_symbol = self._check_capture_before_push(
                self._cchess, human_move)
            exp_rc, exp_bc = self._get_cchess_piece_counts()
            if is_capture and captured_symbol:
                ptype = self._CChess_SYMBOL_TO_TYPE.get(captured_symbol)
                if ptype:
                    if captured_symbol.isupper():
                        exp_rc[ptype] = max(0, exp_rc.get(ptype, 0) - 1)
                    else:
                        exp_bc[ptype] = max(0, exp_bc.get(ptype, 0) - 1)
            exp_red = sum(exp_rc.values())
            exp_black = sum(exp_bc.values())
            valid, reason = VisionController._validate_piece_count(
                board_now, exp_red, exp_black, exp_rc, exp_bc)
            if not valid:
                print(f"[后验] 棋子数与走法不符: {reason}")
                self._vis.print_board(board_now)
                human_move = None
                continue

            # 通过所有校验，推入 cchess
            self._cchess.push(cchess.Move.from_uci(human_move))
            chinese = self._cchess.move_to_notation(cchess.Move.from_uci(human_move))
            print(f"[人类] {chinese} ({human_move})")

            # 记录人类走法（用于悔棋）
            captured_label = self._symbol_to_label(captured_symbol) if is_capture else None
            self._move_history.append(MoveRecord(
                human_move,
                is_capture=is_capture, captured_piece=captured_label))

        self._situation += f" {human_move}"
        self._fen_before_human = self._cchess.fen()

        if self._check_game_over():
            return

        self._last_board = board_now
        red, black = self._vis.count_pieces(board_now)
        print(f"[棋子数] 红:{red} 黑:{black}")

        self._state = self.State.ENGINE_TURN

    def _handle_move_recognition_failure(self, human_move):
        """处理识别失败：区分'与引擎上步相同'和'真的没识别好'"""
        if self._move_history and human_move == self._move_history[-1].uci:
            print("走法与引擎上一步相同，拿回棋子重新拍摄")
            target = human_move[2:] + human_move[:2]
            self._p.send(f"棋子移动:"
                         f"{self._vis.uci_to_pixel_public(human_move[2:])},"
                         f"{self._vis.uci_to_pixel_public(human_move[:2])}")
            self._p.wait_for("玩家落子完成")
            self._last_board = None
            while self._last_board is None:
                if self._p.shutdown_requested():
                    return None
                self._last_board = self._vis.capture_stable_board()
                if self._last_board is None:
                    time.sleep(0.5)
            self._p.send("请玩家落子")
            self._p.wait_for("玩家落子完成")
            return None

        self._p.send("走法无效:请重新落子")
        self._p.wait_for("玩家落子完成")
        return None

    def _handle_illegal_move(self, human_move):
        """处理非法走法：区分'识别错误'和'真的走错'"""
        if self._move_history and human_move == self._move_history[-1].uci:
            print("走法与引擎上一步相同，拿回棋子重新拍摄")
            target = human_move[2:] + human_move[:2]
            self._p.send(f"棋子移动:"
                         f"{self._vis.uci_to_pixel_public(human_move[2:])},"
                         f"{self._vis.uci_to_pixel_public(human_move[:2])}")
            self._p.wait_for("玩家落子完成")
            self._last_board = None
            while self._last_board is None:
                if self._p.shutdown_requested():
                    return None
                self._last_board = self._vis.capture_stable_board()
                if self._last_board is None:
                    time.sleep(0.5)
            self._p.send("请玩家落子")
            self._p.wait_for("玩家落子完成")
            return None

        # 真的走错了，让机器人拿回棋子
        target = human_move[2:] + human_move[:2]
        self._p.send(f"棋子移动:{self._vis.uci_to_pixel_public(human_move[2:])},"
                     f"{self._vis.uci_to_pixel_public(human_move[:2])}")
        self._p.send("请玩家落子")
        self._p.send("文字：不能这样走呀")
        self._p.wait_for("玩家落子完成")
        return None

    # ------------------------------------------------------------------
    def _handle_engine_turn(self):
        current_frame = self._vis._stream.get_latest_warped()

        try:
            engine_move = self._engine.get_best_move(
                self._situation.strip(), self._engine_fen
            )
        except RuntimeError as e:
            print(f"[错误] {e}，尝试恢复引擎...")
            if self._engine.restart(self._situation.strip(), self._engine_fen):
                engine_move = self._engine.get_best_move(
                    self._situation.strip(), self._engine_fen)
            else:
                print("[严重] 引擎恢复失败")
                self._state = self.State.GAME_OVER
                return

        if not engine_move:
            print("[警告] 引擎返回空走法")
            self._state = self.State.GAME_OVER
            return

        # 校验引擎走法合法性（cchess 状态可能被错误的人方走法污染）
        if not self._is_move_legal(engine_move):
            print(f"[严重] 引擎走法 {engine_move} 不合法，尝试摄像头恢复...")
            if self._resync_from_camera():
                self._engine.restart("", self._engine_fen)
                engine_move = self._engine.get_best_move("", self._engine_fen)
                if not engine_move or not self._is_move_legal(engine_move):
                    print(f"[严重] 恢复后引擎走法仍不合法: {engine_move}")
                    self._state = self.State.GAME_OVER
                    return
                print(f"[恢复] 引擎新走法: {engine_move}")
            else:
                print("[严重] 摄像头恢复失败，终止对局")
                self._state = self.State.GAME_OVER
                return

        # 检测吃子（在 push 之前）
        is_capture, captured_symbol = self._check_capture_before_push(
            self._cchess, engine_move)

        # 更新引擎走法到 cchess
        self._push_engine_move(engine_move)
        self._situation += f" {engine_move}"
        chinese = self._cchess.move_to_notation(cchess.Move.from_uci(engine_move))
        print(f"[引擎] {chinese} ({engine_move})")

        if self._cchess.is_check():
            print("[将军] 机器人将军！")
            self._p.send("文字:将军")

        # 吃子 → 先从棋盘移除被吃棋子
        waste_xy = None
        captured_label = None
        if is_capture:
            dst_uci = engine_move[2:]
            _, dst_xy = self._vis.detect_piece_at(dst_uci, current_frame)
            waste_xy = self._get_waste_xy()
            captured_label = self._symbol_to_label(captured_symbol)
            self._use_chess_count -= 1

            self._p.send(f"棋子移动:{dst_xy},{waste_xy}")
            print(f"[提示] 引擎吃子！({captured_symbol})")

        # 发送引擎走法坐标
        src_uci = engine_move[:2]
        dst_uci = engine_move[2:]
        _, src_xy = self._vis.detect_piece_at(src_uci, current_frame)
        dst_pixel = self._vis.uci_to_pixel_public(dst_uci)

        self._p.send(f"棋子移动:{src_xy},{dst_pixel}")
        self._p.wait_for("运动完成")

        self._move_history.append(MoveRecord(
            engine_move, src_xy, dst_pixel,
            is_capture=is_capture, captured_piece=captured_label,
            waste_xy=waste_xy))

        if self._check_game_over():
            return

        self._last_board = self._vis.apply_uci_move(self._last_board, engine_move)
        self._state = self.State.HUMAN_TURN

    # ------------------------------------------------------------------
    def _handle_game_over(self):
        print("[状态] 等待新对局...")
        self._state = self.State.READY

    def _handle_auto_setup(self):
        """一键摆盘：分批摆放 + 逐格校验，循环直到整盘匹配标准 FEN。"""
        print("[摆盘] 开始一键摆盘...")

        target_pieces = Config.fen_to_uci_pieces(self._setup_fen, self._human_is_red)
        BATCH_SIZE = 3
        MAX_RETRIES = 15

        for retry in range(MAX_RETRIES):
            # 每轮重试：先检查棋盘上还缺哪些棋子
            time.sleep(0.5)
            frame = self._vis._stream.get_latest_warped()
            if frame is not None:
                board = self._vis.recognize_board(frame.copy())
            else:
                board = None

            if board and self._validate_against_expected(board, self._setup_fen):
                print("[摆盘] 整盘校验通过！")
                self._p.send("文字:摆盘通过")
                self._state = self.State.READY
                return

            # 构建待摆放列表：只放棋盘上缺失或类型不匹配的棋子
            pending = []
            for uci, label in target_pieces:
                row = 9 - int(uci[1])
                col = ord(uci[0]) - ord('a')
                if board is None or board[row][col] != label:
                    pending.append((uci, label))

            if not pending:
                print("[摆盘] 棋盘已完整但校验未通过，继续重试...")
                continue

            print(f"[摆盘] 第{retry+1}轮: 还需摆放{len(pending)}枚棋子")

            round_count = 0
            while pending and round_count < 10:
                round_count += 1

                # 识别两个废棋区
                human_pieces = self._vis.recognize_waste_area(VisionController.HUMAN_WASTE_ROI)
                robot_pieces = self._vis.recognize_waste_area(VisionController.ROBOT_WASTE_ROI)
                all_waste = human_pieces + robot_pieces
                print(f"[摆盘]  第{round_count}批: 待摆放{len(pending)}枚, "
                      f"废棋区{len(human_pieces)}+{len(robot_pieces)}={len(all_waste)}枚")

                # 匹配本轮棋子
                used = [False] * len(all_waste)
                batch = []
                still_pending = []

                for uci, label in pending:
                    if len(batch) >= BATCH_SIZE:
                        still_pending.append((uci, label))
                        continue

                    target_pixel = self._vis.uci_to_pixel_public(uci)
                    found = None
                    for i, (w_label, wx, wy) in enumerate(all_waste):
                        if not used[i] and w_label == label:
                            found = (wx, wy)
                            used[i] = True
                            break
                    if found:
                        batch.append((uci, label, found, target_pixel))
                    else:
                        still_pending.append((uci, label))

                pending = still_pending

                if not batch:
                    print("[摆盘]  无法匹配更多棋子")
                    break

                # 连续发送本批指令，然后等待
                print(f"[摆盘]  发送 {len(batch)} 条指令...")
                for _, _, src_xy, dst_xy in batch:
                    self._p.send(f"棋子移动:{src_xy},{dst_xy}")
                self._p.wait_for("运动完成")
                time.sleep(0.3)

                # 整盘识别校验本批
                batch_frame = self._vis._stream.get_latest_warped()
                batch_board = self._vis.recognize_board(batch_frame.copy()) if batch_frame is not None else None

                for uci, label, src_xy, dst_xy in batch:
                    if batch_board is not None:
                        r = 9 - int(uci[1])
                        c = ord(uci[0]) - ord('a')
                        cell = batch_board[r][c]
                    else:
                        cell = None

                    if cell is None:
                        print(f"[摆盘]  {uci}({label}) 棋盘格为空，重新入队")
                        pending.append((uci, label))
                    elif cell == label:
                        print(f"[摆盘]  {uci}({label}) 校验通过 ✓")
                    else:
                        print(f"[摆盘]  {uci} 类型不匹配 (期望{label}, 检测到{cell})，移回废棋区")
                        roi = (VisionController.HUMAN_WASTE_ROI if cell.startswith('R_')
                               else VisionController.ROBOT_WASTE_ROI)
                        empty_spot = self._find_empty_spot_in_waste(roi)
                        self._p.send(f"棋子移动:{dst_xy},{empty_spot}")
                        self._p.wait_for("运动完成")
                        pending.append((uci, label))

            # 内层循环结束，回到外层校验

        # 超过最大重试，用 recognize_board 做最终校验
        print("[摆盘] 超过最大重试次数")
        frame = self._vis._stream.get_latest_warped()
        final_board = self._vis.recognize_board(frame.copy()) if frame is not None else None
        if final_board and self._validate_against_expected(final_board, self._setup_fen):
            self._p.send("文字:摆盘通过")
        else:
            self._p.send("文字:摆盘未通过，请检查")
        self._state = self.State.READY

    def _handle_undo(self):
        result = self.request_undo()
        if result is None:
            self._p.send("请玩家落子")
            self._p.send("文字:请落子")
            self._state = self.State.HUMAN_TURN
            return

        m_engine, m_human = result

        human_dst = self._vis.uci_to_pixel_public(m_human.uci[2:])
        human_src = self._vis.uci_to_pixel_public(m_human.uci[:2])

        cmds = []

        # ① 引擎棋子移回：dst → src
        cmds.append(f"棋子移动:{m_engine.dst_xy},{m_engine.src_xy}")

        # ② + ③ 人类棋子移回
        # 引擎吃了人类刚走的棋子 → 直接从废棋区移回人类原位置（省去一次搬动）
        # 引擎吃了其他棋子     → 先恢复被吃棋子到引擎 dst，再移回人类棋子
        if m_engine.is_capture and m_engine.waste_xy:
            if m_engine.uci[2:] == m_human.uci[2:]:
                cmds.append(f"棋子移动:{m_engine.waste_xy},{human_src}")
            else:
                cmds.append(f"棋子移动:{m_engine.waste_xy},{m_engine.dst_xy}")
                cmds.append(f"棋子移动:{human_dst},{human_src}")
        else:
            cmds.append(f"棋子移动:{human_dst},{human_src}")

        # ④ 若人类吃子：从人类废棋区 → 人类 dst
        if m_human.is_capture and m_human.captured_piece:
            xy = self._find_piece_in_human_waste(m_human.captured_piece)
            cmds.append(f"棋子移动:{xy},{human_dst}")

        # 连续发送，不等待
        for cmd in cmds:
            self._p.send(cmd)

        self._p.send("请玩家落子")
        self._p.send("文字:请落子")
        self._state = self.State.HUMAN_TURN

    # ==================================================================
    # 辅助方法
    # ==================================================================
    def _reset_game_state(self):
        self._cchess = None
        self._situation = ""
        self._initial_fen = ""
        self._move_history.clear()
        self._use_chess_count = 16
        self._last_board = None

    def _is_move_legal(self, uci_move: str) -> bool:
        """校验走法在 cchess 中是否合法（不改变局面）"""
        try:
            return self._cchess.is_legal(cchess.Move.from_uci(uci_move))
        except Exception:
            return False

    def _push_engine_move(self, uci_move: str):
        try:
            self._cchess.push(cchess.Move.from_uci(uci_move))
        except Exception as e:
            print(f"[cchess] push 引擎走法失败: {e}")

    def _validate_against_expected(self, board: List[List[str]],
                                     expected_fen: str) -> bool:
        """将识别到的棋盘与预期 FEN 逐格对比，必须完全一致。"""
        expected_board = VisionController.board_from_fen(expected_fen)
        mismatches = []
        for r in range(10):
            for c in range(9):
                exp = expected_board[r][c]
                got = board[r][c]
                if exp != got:
                    mismatches.append((r, c, exp, got))

        if not mismatches:
            print("[校验] 识别结果与预期 FEN 完全一致 ✓")
            return True

        print(f"[校验] 识别 vs 预期 FEN：{len(mismatches)} 处差异")
        for r, c, exp, got in mismatches:
            exp_str = Config.CHINESE_MAP.get(exp, "?") if exp else "空"
            got_str = Config.CHINESE_MAP.get(got, "?") if got else "空"
            print(f"  ({r},{c}): 预期={exp_str}  识别={got_str}")

        print(f"[警告] 差异({len(mismatches)}处)，需要重新识别")
        return False

    def _resync_from_camera(self) -> bool:
        """从摄像头重新捕获局面，重建 cchess 和引擎状态"""
        print("[恢复] 正在从摄像头重建局面...")
        board = self._vis.capture_stable_board()
        if board is None:
            print("[恢复] 摄像头捕获失败")
            return False

        side = "b" if self._human_is_red else "w"
        fen = self._vis.construct_fen(board, side_to_move=side)
        valid, err = Config.validate_fen(fen)
        if not valid:
            print(f"[恢复] 重建的 FEN 不合法: {err}")
            return False

        try:
            self._cchess = cchess.Board(fen)
        except Exception as e:
            print(f"[恢复] cchess 解析 FEN 失败: {e}")
            return False

        self._initial_fen = fen
        self._situation = ""
        self._last_board = board
        print(f"[恢复] 局面已重建，引擎方={side}")
        VisionController.print_board(board)
        return True

    def _check_game_over(self) -> bool:
        if self._cchess.is_game_over():
            # turn 是走子方：刚刚被将死的一方是刚走完的一方
            winner = "人类" if self._cchess.turn != self._human_is_red else "机器人"
            self._p.send("文字:"f"{winner}获胜")
            print("\n=============================")
            print(f"  游戏结束！{winner}获胜。")
            print("=============================")
            self._state = self.State.GAME_OVER
            return True
        return False

    def _get_waste_xy(self) -> Tuple[int, int]:
        """机器人废棋区坐标（右侧单列，从上往下依次摆放）"""
        n = 16 - self._use_chess_count
        waste_x = 660
        waste_y = 18 + n * 52
        return (waste_x, waste_y)

    @staticmethod
    def _get_human_waste_xy() -> Tuple[int, int]:
        """人类废棋区坐标（左侧单列，从上往下）"""
        # HUMAN_WASTE_ROI = (10, 0, 103, 625)，x 取中间约 56
        waste_x = 56
        waste_y = 18
        return (waste_x, waste_y)

    @staticmethod
    def _check_capture_before_push(board: cchess.Board,
                                   uci_move: str) -> Tuple[bool, Optional[str]]:
        try:
            move = cchess.Move.from_uci(uci_move)
        except Exception:
            return False, None
        target = board.piece_at(move.to_square)
        if target is not None:
            return True, target.symbol()
        return False, None

    def _get_cchess_piece_counts(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """从 cchess 权威状态读取棋子类型数量。"""
        rc: Dict[str, int] = {}
        bc: Dict[str, int] = {}
        for sq in range(90):
            piece = self._cchess.piece_at(sq)
            if piece is None:
                continue
            symbol = piece.symbol()
            ptype = self._CChess_SYMBOL_TO_TYPE.get(symbol)
            if not ptype:
                continue
            if symbol.isupper():
                rc[ptype] = rc.get(ptype, 0) + 1
            else:
                bc[ptype] = bc.get(ptype, 0) + 1
        return rc, bc

    def _compute_expected_counts_after_move(
            self, last_board: List[List[str]],
            is_capture: bool, captured_symbol: Optional[str]
            ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """根据走子前的棋盘和吃子信息，推算走子后的预期棋子类型数量。"""
        rc, bc = VisionController.count_pieces_by_type(last_board)
        if is_capture and captured_symbol:
            ptype = self._CChess_SYMBOL_TO_TYPE.get(captured_symbol)
            if ptype:
                if captured_symbol.isupper():
                    rc[ptype] = max(0, rc.get(ptype, 0) - 1)
                else:
                    bc[ptype] = max(0, bc.get(ptype, 0) - 1)
        return rc, bc

    def _symbol_to_label(self, symbol: str) -> Optional[str]:
        """cchess 符号 → 棋子标签，如 'R' → 'R_Car', 'r' → 'B_Car'"""
        ptype = self._CChess_SYMBOL_TO_TYPE.get(symbol)
        if not ptype:
            return None
        prefix = 'R_' if symbol.isupper() else 'B_'
        return prefix + ptype

    # ==================================================================
    # 悔棋接口
    # ==================================================================
    def _find_piece_in_human_waste(self, piece_label: str) -> Tuple[int, int]:
        """从人类废棋区（左侧）通过视觉识别找到指定棋子，返回 warped 坐标。"""
        pieces = self._vis.recognize_waste_area(VisionController.HUMAN_WASTE_ROI)
        for label, x, y in pieces:
            if label == piece_label:
                print(f"[悔棋] 在人类废棋区找到 {piece_label} @ ({x}, {y})")
                return (x, y)
        # 找不到则返回废棋区中心默认坐标
        print(f"[悔棋] 未在人类废棋区找到 {piece_label}，使用默认坐标")
        return (52, 312)

    def _find_empty_spot_in_waste(self, roi_rect: Tuple[int, int, int, int]
                                  ) -> Tuple[int, int]:
        """在废棋区中找到第一个可容纳一枚棋子的空白位置，返回 warped 坐标。

        棋子半径 30px，间隙 >= 60px 即可放入。不会超出 ROI 矩形框边缘。
        """
        PIECE_R = 30  # 棋子半径
        PIECE_D = PIECE_R * 2  # 棋子直径

        pieces = self._vis.recognize_waste_area(roi_rect)
        x1, y1, x2, y2 = roi_rect
        cx = (x1 + x2) // 2

        # 边界约束：棋子中心至少离边缘 30px
        min_x = x1 + PIECE_R
        max_x = x2 - PIECE_R
        min_y = y1 + PIECE_R
        max_y = y2 - PIECE_R

        cx = max(min_x, min(max_x, cx))

        if not pieces:
            return (cx, min_y)

        sorted_by_y = sorted(pieces, key=lambda p: p[2])

        # 顶部到第一颗之间
        if sorted_by_y[0][2] - y1 >= PIECE_D + PIECE_R:
            return (cx, min_y)

        # 棋子之间的间隙
        for i in range(len(sorted_by_y) - 1):
            gap = sorted_by_y[i + 1][2] - sorted_by_y[i][2]
            if gap >= PIECE_D:
                mid_y = (sorted_by_y[i][2] + sorted_by_y[i + 1][2]) // 2
                return (cx, max(min_y, min(max_y, mid_y)))

        # 最后一颗到底部之间
        if y2 - sorted_by_y[-1][2] >= PIECE_D + PIECE_R:
            return (cx, sorted_by_y[-1][2] + PIECE_R)

        # 都没空间，放最底部（钳位）
        return (cx, max_y)

    def request_undo(self) -> Optional[Tuple[MoveRecord, MoveRecord]]:
        """撤销两步（人+引擎），只负责局面恢复，返回弹出的一对 MoveRecord。"""
        if len(self._move_history) < 2:
            print("[悔棋] 走法数不足，无法悔棋")
            return None

        m_engine = self._move_history.pop()
        m_human = self._move_history.pop()

        # --- 1. 反向更新 _last_board ---
        # 先撤销引擎走法
        rev_eng = m_engine.uci[2:] + m_engine.uci[:2]
        self._last_board = self._vis.apply_uci_move(self._last_board, rev_eng)
        if m_engine.is_capture and m_engine.captured_piece:
            col = ord(m_engine.uci[2]) - ord('a')
            row = 9 - int(m_engine.uci[3])
            self._last_board[row][col] = m_engine.captured_piece

        # 再撤销人类走法
        rev_human = m_human.uci[2:] + m_human.uci[:2]
        self._last_board = self._vis.apply_uci_move(self._last_board, rev_human)
        if m_human.is_capture and m_human.captured_piece:
            col = ord(m_human.uci[2]) - ord('a')
            row = 9 - int(m_human.uci[3])
            self._last_board[row][col] = m_human.captured_piece

        # --- 2. 更新 _situation ---
        parts = self._situation.strip().split()
        moves_before = " ".join(parts[:-2]) if len(parts) >= 2 else ""
        self._situation = moves_before

        # --- 3. 恢复 _use_chess_count ---
        if m_engine.is_capture:
            self._use_chess_count += 1
        if m_human.is_capture:
            self._use_chess_count += 1

        # --- 4. 重建 cchess 局面 ---
        if self._cchess_fen:
            self._cchess = cchess.Board(self._cchess_fen)
        else:
            self._cchess = cchess.Board()
        if self._situation.strip():
            for m in self._situation.strip().split():
                self._cchess.push(cchess.Move.from_uci(m))

        # --- 5. 重启引擎 ---
        self._engine.restart(self._situation.strip(), self._engine_fen)

        print(f"[悔棋] 已撤销引擎 {m_engine.uci} 和人类 {m_human.uci}")
        return (m_engine, m_human)


# ============================================================================
# 入口
# ============================================================================

def main():
    """主函数 —— 组装所有组件并启动对弈循环"""
    print("=" * 60)
    print("  象棋机器人自动对弈系统 v5.0")
    print("=" * 60)

    # --- 1. 环境初始化 ---
    print("\n[系统] 正在初始化...")

    try:
        yolo_model = YOLO(Config.MODEL_PATH)
        print("[系统] YOLO 模型加载完成")
    except Exception as e:
        print(f"[严重] 模型加载失败: {e}")
        return

    engine = EngineController()
    try:
        engine.start()
    except Exception as e:
        print(f"[严重] 引擎启动失败: {e}")
        return

    video_stream = VideoStreamThread(0)
    video_stream.start()
    time.sleep(0.5)

    proto = ProtocolClient()
    vision = VisionController(yolo_model, video_stream)
    game = GameController(proto, engine, vision)

    # --- 2. 连接服务器并注册 ---
    try:
        proto.connect()
        proto.send("REGISTER:象棋引擎")
        print("[系统] 已向服务器注册")
    except Exception as e:
        print(f"[严重] 服务器连接失败: {e}")
        engine.stop()
        video_stream.stop()
        return

    # --- 3. 引擎握手 ---
    if not engine.uci_handshake():
        print("[严重] 引擎握手失败")
        engine.stop()
        video_stream.stop()
        return

    print("[系统] 初始化完成，进入对弈循环。")

    # --- 4. 主循环 ---
    try:
        game.run()
    except KeyboardInterrupt:
        print("\n[系统] 用户中断")
    except Exception as e:
        print(f"[错误] 未捕获异常: {e}")
        import traceback
        traceback.print_exc()

    # --- 5. 资源释放 ---
    print("[清理] 正在释放资源...")
    engine.stop()
    proto.disconnect()
    video_stream.stop()
    print("[系统] 已退出。")


if __name__ == "__main__":
    main()
