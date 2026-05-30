import random
import re
import time
import traceback
from collections import Counter

import openai
from openai import OpenAI

from agents.base import ChatCallback, GooseAgent, GooseAgentMessage, GooseAgentResult, PlannerAgent
from goose_game.environment import GooseEnvironment, PlannerEnvironment
from goose_game.models import Direction

# Tunables
MOVE_VOTES = 3        # majority vote on the brittle "which direction" call
YESNO_VOTES = 1       # raise if completion checks prove flaky
HOLD_PATIENCE = 6     # turns to hold a button before treating it as the wrong one
MAX_RETRIES = 8

# Words in a command that mean "do not move this turn" (honk).
# Deliberately excludes "stand" so "Reach/Stand on the button" still navigates.
# holding is phrased with "stay"/"hold".
HOLD_KEYWORDS = ("honk", "wait", "stay", "hold", "remain", "do nothing")

RETRYABLE = (openai.InternalServerError, openai.APITimeoutError, openai.APIConnectionError)


def get_model_answer(client, model, system_prompt, user_prompt, max_retries=MAX_RETRIES):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(model=model, messages=messages)
            if resp.choices and resp.choices[0].message.content:
                return resp.choices[0].message.content
            # empty completion: fall through to backoff
        except openai.RateLimitError:
            time.sleep(60)
            continue
        except RETRYABLE:
            traceback.print_exc()
        except Exception:
            traceback.print_exc()
            raise  # genuine errors (400s, auth, bad args) still surface
        time.sleep(min(2 ** attempt, 60) + random.uniform(0, 1))
    raise RuntimeError(f"get_model_answer failed after {max_retries} retries")


def parse_move(text):
    """Return a Direction, the string 'HONK', or None if unparseable."""
    t = text.strip().lower()
    if re.search(r"\b(honk|stay|wait|hold|remain)\b", t):
        return "HONK"
    for word, direction in (("up", Direction.UP), ("down", Direction.DOWN),
                            ("left", Direction.LEFT), ("right", Direction.RIGHT)):
        if re.search(rf"\b{word}\b", t):
            return direction
    return None


def parse_yes(text):
    """True only on a clear YES; anything ambiguous is treated as NO (conservative:
    do not declare a sub-goal complete, do not release a hold, unless sure)."""
    return bool(re.search(r"\byes\b", text.strip().lower()))


def vote_yes(client, model, system_prompt, user_prompt, n=YESNO_VOTES):
    yes = sum(parse_yes(get_model_answer(client, model, system_prompt, user_prompt)) for _ in range(n))
    return yes * 2 > n


def subgoal_kind(text):
    t = text.lower()
    if "hold" in t or ("stay" in t and "button" in t):
        return "HOLD"
    if "honk" in t and "goal" in t:
        return "HONK_GOAL"
    if "goal" in t:
        return "GOAL"
    if "button" in t:
        return "BUTTON"
    return "OTHER"


# Prompts
_LEGEND = ("Map legend: # wall, . empty, * goal, @ button, $ closed door, "
           "/ open door, ? unknown, X goose 1, Y goose 2.\n")

GOOSE_MOVE_SYSTEM = (
    "You are a goose in a grid puzzle. Output exactly one word and nothing else: "
    "UP, DOWN, LEFT, RIGHT, or HONK.\n"
    "UP = north, DOWN = south, LEFT = west, RIGHT = east. HONK = stay in place.\n"
    + _LEGEND +
    "You may step onto empty squares, the goal, buttons, and open doors. "
    "You cannot step into walls, closed doors, or unknown squares.\n"
)
GOOSE_MOVE_PROMPT = (
    "Map:\n{}\n\n"
    "{}\n"  # self-position anchor from the environment
    "Look at your own square on the map using the position given above. If it shows *, you are on the goal.\n"
    "Your instruction: {}\n"
    "Find the target of your instruction on the map. Take the single step (UP/DOWN/LEFT/RIGHT) that moves "
    "you closer to it, moving along whichever axis you are furthest from it on, and only into a square that "
    "is not a wall, a closed door, or unknown. If you are already on the target, answer HONK.\n"
    "Answer with one word only.\n"
)

GOOSE_REPORT_SYSTEM = (
    "You are a goose in a grid puzzle, reporting what you see to your planner.\n"
    + _LEGEND +
    "Describe everything relative to yourself: compass direction and distance in squares. "
    "Never use numeric coordinates.\n"
)
GOOSE_REPORT_PROMPT = (
    "Map:\n{}\n\n"
    "{}\n"  # self-position anchor from the environment
    "Use the position given above to find yourself on the map. If your own square shows *, you are standing "
    "on the goal.\n"
    "Report exactly in this shape, describing everything relative to yourself, with NO coordinates:\n"
    "\"I am standing on [the goal / an ordinary square]. North: [what is in the square immediately north]. "
    "South: [...]. East: [...]. West: [...]. Also visible: [each button, door, the goal, and the other "
    "goose you can see, with direction and distance].\"\n"
    "Note: if you are standing on a button you will see your own marker, not @, so just call your square "
    "ordinary.\n"
)

PLANNER_SYSTEM = (
    "You coordinate two geese, goose_1 and goose_2, in a grid puzzle. The level is solved when both stand "
    "on the goal (*) and honk.\n"
    "World rules: a button opens a door only while a goose stands on it; a goose cannot hold a door open "
    "for itself; some buttons do nothing; walls and closed doors block movement. Which button opens which "
    "door is not given and must be worked out by trying.\n"
    "A goose standing on a button cannot see it (its own marker hides the @), so it will report an "
    "ordinary square.\n"
)
PLANNER_SELECT_PROMPT = (
    "Level objective: {}\n\n"
    "Progress so far:\n{}\n\n"
    "Latest report from {}: {}\n"
    "Latest report from the other goose: {}\n\n"
    "By default a goose should reach the goal and then honk on it - this is almost always the correct "
    "sub-goal. Only give a goose a helping sub-goal (reaching or holding a button) when a CLOSED door blocks "
    "the OTHER goose's path to the goal and a button is available to open it. Never send a goose to chase or "
    "reach the other goose.\n"
    "Choose the single next sub-goal for {}. Output one short command, phrased relative to the goose, and "
    "nothing else. Use 'Reach ...' for moving towards something and 'Stay/Hold ...' for staying put. "
    "Examples:\n"
    "  Reach the goal.\n"
    "  Honk and wait on the goal.\n"
    "  Reach the button to the south.\n"
    "  Stay on the button to hold the door for the other goose.\n"
    "  Go through the open door to the east.\n"
    "Pick exactly one next action; do not chain actions.\n"
)

COMPLETE_GOAL_PROMPT = (
    "A goose was told: \"{}\"\n"
    "Its latest report: {}\n"
    "Is the goose now standing on the goal? Answer YES or NO only.\n"
)
COMPLETE_BUTTON_PROMPT = (
    "A goose was heading for a button: \"{}\"\n"
    "Its earlier report: {}\n"
    "Its latest report: {}\n"
    "A goose on a button cannot see it. If the button it was approaching is no longer in an immediately "
    "adjacent square, it has most likely stepped onto it. Has the goose reached the button? YES or NO only.\n"
)
COMPLETE_HOLD_PROMPT = (
    "One goose is holding a button to open a door for the other goose.\n"
    "The other goose's latest report: {}\n"
    "Has the other goose reached the goal, so the button no longer needs holding? YES or NO only.\n"
)
COMPLETE_GENERIC_PROMPT = (
    "A goose was told: \"{}\"\n"
    "Its latest report: {}\n"
    "Has this instruction been accomplished? Answer YES or NO only.\n"
)


# Goose
class GooseAgentImpl(GooseAgent):
    def __init__(self, client: OpenAI, used_model: str, env: GooseEnvironment, append_to_chat: ChatCallback) -> None:
        super().__init__(client, used_model, env, append_to_chat)
        self._marker = "X" if env.goose_id.endswith("1") else "Y"
        self.turn_counter = 0
        self._append_to_chat(f"Initialized {env.goose_id}.")

    def on_call(self, message: GooseAgentMessage) -> GooseAgentResult:
        cmd = message.description
        self._append_to_chat(f"Command: {cmd}")
        self._act(cmd, self._anchor())  # anchor reflects position BEFORE the move

        # Recompute the anchor so the report reflects the post-move position.
        report = get_model_answer(self._client, self._used_model,
                                  GOOSE_REPORT_SYSTEM,
                                  GOOSE_REPORT_PROMPT.format(self._env.describe_state(), self._anchor()))
        self._append_to_chat(f"Turn {self.turn_counter}. Report: {report}")
        self.turn_counter += 1
        return GooseAgentResult(output=report)

    def _anchor(self) -> str:
        """self-location from the environment"""
        positions = self._env.visible_goose_positions()
        me = positions.get(self._env.goose_id)
        if me is None:
            return "Your own position is unavailable; find your marker on the map."
        parts = [f"You are the marker {self._marker}, at row {me[0]}, column {me[1]} "
                 f"(rows are counted from the top, columns from the left)."]
        for gid, pos in positions.items():
            if gid != self._env.goose_id:
                other = "X" if gid.endswith("1") else "Y"
                parts.append(f"The other goose ({other}) is at row {pos[0]}, column {pos[1]}.")
        return " ".join(parts)

    def _act(self, cmd: str, anchor: str) -> None:
        if any(k in cmd.lower() for k in HOLD_KEYWORDS):
            self._honk()
            return
        move = self._decide_move(cmd, anchor)
        if move is None or move == "HONK":
            self._honk()
        else:
            self._env.move(move)  # no-op (returns False) if blocked or out of budget

    def _decide_move(self, cmd: str, anchor: str):
        """Majority vote over MOVE_VOTES samples."""
        votes = []
        for _ in range(MOVE_VOTES):
            ans = get_model_answer(self._client, self._used_model,
                                   GOOSE_MOVE_SYSTEM,
                                   GOOSE_MOVE_PROMPT.format(self._env.describe_state(), anchor, cmd))
            self._append_to_chat(f"  move sample: {ans.strip()}")
            m = parse_move(ans)
            if m is not None:
                votes.append(m)
        if not votes:
            return "HONK"
        return Counter(votes).most_common(1)[0][0]

    def _honk(self) -> None:
        if self._env.can_take_counted_action():
            self._env.honk(1)


# Planner
class PlannerAgentImpl(PlannerAgent):
    def __init__(
        self,
        client: OpenAI,
        used_model: str,
        env: PlannerEnvironment,
        agents: dict[str, GooseAgent],
        append_to_chat: ChatCallback,
    ) -> None:
        super().__init__(client, used_model, env, agents, append_to_chat)
        self._subgoal: dict[str, str | None] = {gid: None for gid in self._agents}
        self._last_reports: dict[str, str] = {gid: "No report yet." for gid in self._agents}
        self._prev_reports: dict[str, str] = {gid: "No report yet." for gid in self._agents}
        self._hold_turns: dict[str, int] = {gid: 0 for gid in self._agents}
        self._progress: list[str] = []          # append-only plan memory (latching by construction)
        self.turn_counter = 0
        self._append_to_chat(f"Initialized planner for level: {env.level_name}.")

    def step(self) -> None:
        self._append_to_chat("Planner step executed.")
        for goose_id, goose in sorted(self._agents.items()):
            other_id = self._other(goose_id)
            self._ensure_subgoal(goose_id, other_id)

            command = self._subgoal[goose_id]
            self._append_to_chat(f"Calling {goose_id}: {command}")
            result = goose.on_call(GooseAgentMessage(description=command))

            if result.error is not None:
                self._append_to_chat(f"{goose_id} error: {result.error}")
            else:
                self._prev_reports[goose_id] = self._last_reports[goose_id]
                self._last_reports[goose_id] = result.output
                self._append_to_chat(f"{goose_id} report: {result.output}")
        self.turn_counter += 1

    def _other(self, goose_id: str) -> str:
        others = [g for g in self._agents if g != goose_id]
        return others[0] if others else goose_id

    def _ensure_subgoal(self, gid: str, other_id: str) -> None:
        """The gate: keep the committed sub-goal unless it is complete (or a hold has
        run out of patience). Only then re-select."""
        sg = self._subgoal[gid]
        if sg is not None:
            if self._is_complete(gid, other_id, sg):
                self._progress.append(f"{gid} completed: {sg}")
                sg = None
            elif subgoal_kind(sg) == "HOLD":
                self._hold_turns[gid] += 1
                if self._hold_turns[gid] > HOLD_PATIENCE:
                    self._progress.append(
                        f"{gid} held its button for {self._hold_turns[gid]} turns but {other_id} did not "
                        f"reach the goal; that button is probably the wrong one.")
                    sg = None

        if sg is None:
            sg = self._select(gid, other_id)
            self._hold_turns[gid] = 0
            self._append_to_chat(f"{gid} new sub-goal: {sg}")
        self._subgoal[gid] = sg

    def _is_complete(self, gid: str, other_id: str, sg: str) -> bool:
        kind = subgoal_kind(sg)
        if kind == "HONK_GOAL":
            return False  # terminal honking on the goal
        if kind == "GOAL":
            prompt = COMPLETE_GOAL_PROMPT.format(sg, self._last_reports[gid])
        elif kind == "BUTTON":
            prompt = COMPLETE_BUTTON_PROMPT.format(sg, self._prev_reports[gid], self._last_reports[gid])
        elif kind == "HOLD":
            prompt = COMPLETE_HOLD_PROMPT.format(self._last_reports[other_id])
        else:
            prompt = COMPLETE_GENERIC_PROMPT.format(sg, self._last_reports[gid])
        return vote_yes(self._client, self._used_model, PLANNER_SYSTEM, prompt)

    def _select(self, gid: str, other_id: str) -> str:
        progress = "\n".join(self._progress[-6:]) or "Nothing yet."
        cmd = get_model_answer(self._client, self._used_model, PLANNER_SYSTEM,
                               PLANNER_SELECT_PROMPT.format(self._env.task_description,
                                                            progress,
                                                            gid, self._last_reports[gid],
                                                            self._last_reports[other_id],
                                                            gid))
        return cmd.strip().strip('"').splitlines()[0].strip()