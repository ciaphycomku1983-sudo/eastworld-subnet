# The MIT License (MIT)
# Copyright © 2025 Eastworld AI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from __future__ import annotations

import json
import queue
import shlex
import threading
from dataclasses import dataclass
from typing import Iterable

import bittensor as bt


class ManualControlError(Exception):
    """Raised when a manual command cannot be parsed into an action."""


class ManualControlSkip(Exception):
    """Raised to indicate that no action should be taken (e.g. help command)."""


@dataclass
class ManualAction:
    name: str
    arguments: dict


def _parse_scalar(value: str):
    lowered = value.lower()
    if lowered == "none":
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"

    # Try numeric conversions
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            num = float(value)
            if num.is_integer():
                return int(num)
            return num
        return int(value)
    except ValueError:
        pass

    # Try JSON only for structured inputs
    if value.startswith("{") or value.startswith("[") or value.startswith("\""):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

    return value


class ManualController:
    """Simple CLI-driven controller that can override the SeniorAgent's actions."""

    def __init__(self, directions: Iterable[str]):
        self._directions = {direction.lower() for direction in directions}
        self._commands: "queue.Queue[str]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _listen_loop(self):
        bt.logging.info(
            "Manual control enabled. Type 'help' or '?' for instructions. Type 'exit' to return to autonomous mode."
        )
        while not self._stop_event.is_set():
            try:
                raw = input("manual> ")
            except EOFError:
                bt.logging.warning("Manual control input stream closed (EOF); continuing autonomously.")
                break
            except KeyboardInterrupt:
                bt.logging.warning("Manual control interrupted; continuing autonomously.")
                break

            if raw is None:
                continue

            raw = raw.strip()
            if not raw:
                continue

            if raw.lower() in {"exit", "quit"}:
                bt.logging.info("Manual control session ended. Continuing with LLM policy.")
                break

            self._commands.put(raw)

        self._stop_event.set()

    def _drain_command(self) -> str | None:
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return None

    def poll_action(self, synapse, local_action_space) -> ManualAction | None:
        """Attempt to build an action from the queued manual commands."""

        if self._stop_event.is_set():
            return None

        command = self._drain_command()
        while command is not None:
            try:
                return self._parse_command(command, synapse, local_action_space)
            except ManualControlSkip:
                command = self._drain_command()
                continue
            except ManualControlError as exc:
                bt.logging.error(f"Manual command error: {exc}")
                command = self._drain_command()
                continue

        return None

    # ---------------------------------------------------------------------
    # Command parsing helpers
    # ---------------------------------------------------------------------
    def _parse_command(self, raw: str, synapse, local_action_space) -> ManualAction:
        tokens = shlex.split(raw)
        if not tokens:
            raise ManualControlSkip()

        command = tokens[0].lower()
        if command in {"help", "?"}:
            self._print_help(synapse, local_action_space)
            raise ManualControlSkip()

        if command == "list":
            self._print_action_list(synapse, local_action_space)
            raise ManualControlSkip()

        if command in {"move", "m"}:
            return self._parse_move(tokens)

        if command in {"call", "do", "action", "use"}:
            return self._parse_named_action(tokens, synapse, local_action_space)

        if command == "json":
            return self._parse_json_action(raw)

        raise ManualControlError(
            "Unknown command. Type 'help' to see available manual control commands."
        )

    def _parse_move(self, tokens: list[str]) -> ManualAction:
        if len(tokens) < 3:
            raise ManualControlError("Usage: move <direction> <distance>")

        direction = tokens[1].lower()
        if direction not in self._directions:
            valid = ", ".join(sorted(self._directions))
            raise ManualControlError(f"Unknown direction '{direction}'. Valid options: {valid}")

        distance_token = tokens[2]
        try:
            distance = float(distance_token)
        except ValueError:
            raise ManualControlError("Distance must be a number") from None

        if distance <= 0:
            raise ManualControlError("Distance must be greater than zero")

        return ManualAction(
            name="move_in_direction",
            arguments={"direction": direction, "distance": distance},
        )

    def _parse_named_action(self, tokens: list[str], synapse, local_action_space) -> ManualAction:
        if len(tokens) < 2:
            raise ManualControlError("Usage: call <action_name> [key=value ...]")

        action_lookup = self._build_action_lookup(synapse, local_action_space)
        action_name = tokens[1]
        if action_name not in action_lookup:
            available = ", ".join(sorted(action_lookup)) or "(no tools available)"
            raise ManualControlError(
                f"Unknown action '{action_name}'. Available actions: {available}"
            )

        arguments: dict[str, object] = {}
        for token in tokens[2:]:
            if "=" not in token:
                raise ManualControlError(
                    f"Argument '{token}' must be provided in key=value format"
                )
            key, raw_value = token.split("=", 1)
            key = key.strip()
            if not key:
                raise ManualControlError("Argument keys must be non-empty")
            arguments[key] = _parse_scalar(raw_value)

        return ManualAction(name=action_name, arguments=arguments)

    def _parse_json_action(self, raw: str) -> ManualAction:
        _, _, json_payload = raw.partition(" ")
        json_payload = json_payload.strip()
        if not json_payload:
            raise ManualControlError(
                "Usage: json {\"name\": \"<action_name>\", \"arguments\": {...}}"
            )

        try:
            data = json.loads(json_payload)
        except json.JSONDecodeError as exc:
            raise ManualControlError(f"Invalid JSON payload: {exc}") from None

        if not isinstance(data, dict):
            raise ManualControlError("JSON payload must decode to an object")

        name = data.get("name")
        if not isinstance(name, str) or not name:
            raise ManualControlError("JSON payload must include a non-empty 'name'")

        arguments = data.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ManualControlError("'arguments' must be an object if provided")

        return ManualAction(name=name, arguments=arguments)

    def _build_action_lookup(self, synapse, local_action_space) -> dict[str, dict]:
        lookup: dict[str, dict] = {}

        def register(actions: Iterable[dict]):
            for action in actions:
                fn = action.get("function", {})
                name = fn.get("name")
                if name:
                    lookup[name] = fn

        register(synapse.action_space or [])
        register(local_action_space or [])
        return lookup

    def _print_help(self, synapse, local_action_space):
        directions = ", ".join(sorted(self._directions))
        bt.logging.info(
            """
Manual control commands:
  move <direction> <distance>     - Move the robot manually (directions: %s)
  call <action> key=value [...]   - Invoke a supported tool/action with arguments
  json {"name":..., "arguments":...} - Provide a fully-specified action as JSON
  list                            - Show available actions from the current step
  exit                            - Stop manual overrides (LLM will take over)
"""
            % directions
        )
        self._print_action_list(synapse, local_action_space)

    def _print_action_list(self, synapse, local_action_space):
        actions = self._build_action_lookup(synapse, local_action_space)
        if not actions:
            bt.logging.info("No callable actions available from the environment.")
            return

        bt.logging.info("Available actions:")
        for name, fn in sorted(actions.items()):
            description = fn.get("description", "")
            if description:
                bt.logging.info(f"  - {name}: {description}")
            else:
                bt.logging.info(f"  - {name}")
