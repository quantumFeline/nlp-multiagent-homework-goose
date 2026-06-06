import time
import traceback

import openai
from openai import OpenAI

from agents.base import ChatCallback, GooseAgent, GooseAgentMessage, GooseAgentResult, PlannerAgent
from goose_game.environment import GooseEnvironment, PlannerEnvironment
from goose_game.models import Direction

MAP_LEGEND = "# wall  . empty  * goal  @ button  $ closed door  / open door  ? unknown  X goose_1  Y goose_2\n"
PLANNER_MAP_LEGEND = "# wall  . empty  * goal  B button  D closed door  O open door  ? unknown  X goose_1  Y goose_2\n"

SYMBOL_TO_NAME = {
    "=#": "=wall", "=.": "=empty", "=*": "=goal",
    "=@": "=button", "=$": "=closed_door", "=/": "=open_door",
    "=?": "=not yet seen", "=X": "=goose_1", "=Y": "=goose_2",
}

OTHER_GOOSE = {
    "goose_1": "goose_2",
    "goose_2": "goose_1",
}

def expand_report(report: str) -> str:
    """Expand raw map symbols in a structured goose report to unambiguous names."""
    last_line = report.strip().split('\n')[-1]
    for sym, name in SYMBOL_TO_NAME.items():
        last_line = last_line.replace(sym, name)
    return last_line

def expand_map(grid: str) -> str:
    """Replace confusable map symbols with distinct single characters for the planner."""
    return grid.replace("@", "B").replace("$", "D").replace("/", "O")

# No args
PLANNER_SYSTEM_PROMPT = ("You are playing the GOOSE game. The goal of the game is to get all GEESE to the GOAL and have them HONK while standing on their GOAL.\n"
                         "Your role is PLANNER.\n"
                         "PLANNER has two modes: THINK and COMMAND.\n"
                         "When in COMMAND mode, you give orders to the GOOSE plays so that they can solve a task.\n"
                         "When in THINK mode, you must make notes for the future PLANNER so that it knows how to command in the next round.\n")

PLANNER_THINK_PROMPT_GOOSE_1 = ("You are currently in THINK mode. Your task is to pass your knowledge to the next round.\n"
                        "Your previous notes were:\n"
                        "{}\n"
                        "The current state of the game map is:\n"
                        "{}\n"
                        "Map legend: " + PLANNER_MAP_LEGEND + "\n"
                        "The task for this map is:\n"
                        "{}\n"
                        "The last report from goose_1 is: {}\n"
                        "The goose_1 coordinates are: x={}, y={}\n"
                        "The path towards the goal for goose_1 is {}. It was previously {}.\n"
                        "Your notes should reflect it if anything of the following has happened:\n"
                        "- goose_1 has stepped on the goal;\n"
                        "- goose_1 has stepped on the button (include button coordinates);\n"
                        "- goose_1 is holding the button (include button coordinates);\n"
                        "- goose_1 has stepped on the open door;\n"
                        "- a previously BLOCKED path has just become CLEAR;\n"
                        "- a previously CLEAR path has just become BLOCKED.\n"
                        "If none of the above happened, the note must be empty.\n")
# previous notes, state map, task, last report, coordinate_x, coordinate_y, clear? prev clear?

PLANNER_THINK_PROMPT_GOOSE_2 = ("You are currently in THINK mode. Your task is to pass your knowledge to the next round.\n"
                        "Your previous notes were:\n"
                        "{}\n"
                        "The current state of the game map is:\n"
                        "{}\n"
                        "Map legend: " + PLANNER_MAP_LEGEND + "\n"
                        "The task for this map is:\n"
                        "{}\n"
                        "The last report from goose_1 is: {}\n"
                        "The goose_2 coordinates are: {}\n"
                        "The path towards the goal for goose_2 is {}. It was previously {}.\n"
                        "Your notes should reflect it if anything of the following has happened:\n"
                        "- goose_2 has stepped on the goal;\n"
                        "- goose_2 has stepped on the button (include button coordinates);\n"
                        "- goose_2 is holding the button (include button coordinates);\n"
                        "- goose_2 has stepped on the open door;\n"
                        "- a previously BLOCKED path has just become CLEAR;\n"
                        "- a previously CLEAR path has just become BLOCKED.\n"
                        "If none of the above happened, the note must be empty.\n")
# previous notes, state map, task, last report, coordinate_x, coordinate_y, clear?, prev clear?

PLANNER_PROMPT_GOOSE_1 = ("You are currently in COMMAND mode. Your task is to give a COMMAND to goose_1 located at x={}, y={}.\n"
                  "The current known map is:\n"
                  "{}"
                  "Map legend: " + PLANNER_MAP_LEGEND + "\n"
                  "The path towards the goal for goose_1 is {}.\n"
                  "The path towards the goal for goose_2 is {}.\n"
                  "There are following cases:\n"
                  "- If the path is CLEAR for both goose_1 and goose_2, goose_1 is holding the button, and goose_2 has not yet gone through the door, you must command it to wait.\n"
                  "- If the path is CLEAR for both goose_1 and goose_2, goose_1 is holding the button, and goose_2 has already gone through the door, you must command it to proceed towards the goal.\n"
                  "- If the path is CLEAR for both goose_1 and goose_2 and goose_1 is not holding the button, you must command goose_1 to proceed towards the goal.\n"
                  "- If the path is BLOCKED for goose_1 but CLEAR for goose_2, you must command goose_1 to wait.\n"
                  "- If the path is CLEAR for goose_1 but BLOCKED for goose_2, you must command goose_1 to press a button that has not yet been pressed.\n"
                  "- If the path is BLOCKED for goose_1 and BLOCKED for goose_2 and goose_1 sees a button that has not yet been pressed, you must command it to press the button.\n"
                  "- If the path is BLOCKED for goose_1 and BLOCKED for goose_2 and goose_1 does not see a button that has not yet been pressed, you must command it to wait.\n"
                  "Think step-by step, then provide the answer. The last line should be a single sentence that is a command to goose_1.")
# x, y, map, goose 1 clear?, goose 2 clear?

PLANNER_PROMPT_GOOSE_2 = ("You are currently in COMMAND mode. Your task is to give a COMMAND to goose_2 located at x={}, y={}.\n"
                  "The current known map is:\n"
                  "{}"
                  "Map legend: " + PLANNER_MAP_LEGEND + "\n"
                  "The path towards the goal for goose_1 is {}.\n"
                  "The path towards the goal for goose_2 is {}.\n"
                  "There are following cases:\n"
                  "- If the path is CLEAR for both goose_1 and goose_2, goose_2 is holding the button, and goose_1 has not yet gone through the door, you must command it to wait.\n"
                  "- If the path is CLEAR for both goose_1 and goose_2, goose_2 is holding the button, and goose_1 has already gone through the door, you must command it to proceed towards the goal.\n"
                  "- If the path is CLEAR for both goose_1 and goose_2 and goose_2 is not holding the button, you must command goose_2 to proceed towards the goal.\n"
                  "- If the path is BLOCKED for goose_2 but CLEAR for goose_1, you must command goose_2 to wait.\n"
                  "- If the path is CLEAR for goose_2 but BLOCKED for goose_1, you must command goose_2 to press a button that has not yet been pressed.\n"
                  "- If the path is BLOCKED for goose_2 and BLOCKED for goose_1 and goose_2 sees a button that has not yet been pressed, you must command it to press the button.\n"
                  "- If the path is BLOCKED for goose_2 and BLOCKED for goose_1 and goose_2 does not see a button that has not yet been pressed, you must command it to wait.\n"
                  "Think step-by step, then provide the answer. The last line should be a single sentence that is a command to {}.")
# x, y, map, goose 1 clear?, goose 2 clear?

# GOOSE_SYSTEM_PROMPT.format(self._env.goose_id)
GOOSE_SYSTEM_PROMPT = ("You are playing the GOOSE game. The goal of the game is to get all GEESE to the GOAL and have them HONK while standing on their GOAL.\n"
                         "Your role is GOOSE. Your ID is {}.\n"
                         "GOOSE has two modes: OBSERVE and MOVE.\n"
                         "When in OBSERVE mode, you pass useful information to the PLANNER.\n"
                         "When in MOVE mode, you navigate a move in accordance to the plan given to you by the PLANNER.\n")
# goose id

GOOSE_PROMPT = ("You are currently in MOVE mode. Your task is to provide the next MOVE. It can be UP, DOWN, LEFT, RIGHT, or HONK.\n"
                "You HONK if you received a command to stay in place, hold, or wait.\n"
                "The current known map is:\n"
                "{}"
                "Map legend: " + PLANNER_MAP_LEGEND + "\n"
                "Your current instruction is:\n"
                "{}\n"
                "- If you are instructed to press the button, you must navigate towards the button using UP, LEFT, RIGHT, or DOWN.\n"
                "- If you are instructed to proceed towards the goal, you must navigate towards the goal using UP, LEFT, RIGHT, or DOWN.\n"
                "- If you are instructed to wait, you must HONK.\n"
                "Think step-by-step, then output one move. It must be UP, LEFT, RIGHT, DOWN, or HONK.\n")
# map, instruction

# GOOSE_OBS_PROMPT.format(self._env.describe_state(), pos[0], pos[1]))
GOOSE_OBS_PROMPT = ("You are currently in OBSERVE mode. Your task is to provide the PLANNER with the key information.\n"
                           "The current known map is:\n"
                           "{}"
                           "Map legend: " + PLANNER_MAP_LEGEND +
                           "Your last move was {}.\n"
                           "Think step-by step, then provide the answer. The last line of your response should be in the following format:\n"
                           "LEFT of me is [symbol], RIGHT is [symbol], UP is [symbol], DOWN is [symbol]. My last move was [move].\n"
                           "For example:\n"
                           "\"LEFT of me is WALL, RIGHT is EMPTY, UP is BUTTON, DOWN is EMPTY. My last move was UP.\""
                           "\"LEFT of me is EMPTY, RIGHT is GOAL, UP is {}, DOWN is WALL. My last move was HONK.\"")
# map, last move, other goose id

MAP_COMPOSER_SYSTEM_PROMPT = ("You are a MAP COMPOSER.\n"
                              "Your task is to merge maps for a grid-based game.\n"
    "You receive a combined map built so far and a new partial observation, and output an updated combined map.\n"
    "Map symbols: " + MAP_LEGEND +
    "Merging rules:\n"
    "- ? means the cell was not yet observed. Replace ? with the new observation's value if it is known there.\n"
    "- Static elements (#, ., *, @, $, /) once known never revert to ?. Keep them from the combined map even "
    "if the new observation shows ? there.\n"
    "- Goose positions (X, Y) are dynamic. Always take them from the new observation.\n"
    "- If a cell is known in both maps and they disagree on a static element, trust the new observation "
    "(doors can open or close).\n"
    "Output only the updated grid, row by row, with no extra text, explanation, or blank lines.\n"
)

# prompt = MAP_COMPOSER_MERGE_PROMPT.format(self._combined_map, goose_id, raw_observation)
MAP_COMPOSER_MERGE_PROMPT = (
    "Current combined map:\n{}\n\n"
    "New observation from {}:\n{}\n\n"
    "Output the updated combined map.\n"
)

GOAL_ESTIMATOR_SYSTEM_PROMPT = ("You are a GOAL ESTIMATOR for a grid-based game.\n"
    "You are analyzing a game map to determine if a a given GOOSE can reach the goal.\n"
    "Map symbols: " + MAP_LEGEND +
    "A goose is CLEAR if there is at least one path to the goal with no closed doors ($) or walls (#).\n"
    "A goose is BLOCKED if there is a closed door ($) or wall (#) on every path between it and the goal (*).\n"
    "A goose is UNKNOWN if it is not CLEAR and there are unobserved cells (?) between it and the goal that may or may not contain obstacles.\n"
    "Your task is to determine whether there exists a path from the goose to the goal."
)

# GOAL_ESTIMATOR_PROMPT.format(combined_map, goose_id, row, col, goose_id)
GOAL_ESTIMATOR_PROMPT = (
    "Combined map:\n"
    "{}\n"
    "{} is at x={}, y={}.\n"
    "Can {} reach the goal (*)?\n"
    "Think step-by-step, then output one work. It must be CLEAR, BLOCKED, or UNKNOWN.")
# map, goose id, x, y, goose id



class GoalEstimator:
    """Estimates whether a goose can reach the goal given the current combined map."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def estimate(self, goose_id: str, combined_map: str, position: tuple | None) -> str:
        """Return CLEAR, BLOCKED, or UNKNOWN for the given goose."""
        row, col = position if position is not None else (-1, -1)
        answer = get_model_answer(
            self._client, self._model,
            GOAL_ESTIMATOR_SYSTEM_PROMPT,
            GOAL_ESTIMATOR_PROMPT.format(combined_map, goose_id, row, col, goose_id),
        )
        last = answer.strip().split('\n')[-1].strip().upper()
        if last in ("CLEAR", "BLOCKED", "UNKNOWN"):
            return last
        return "UNKNOWN"


class MapComposer:
    """Maintains a combined map built from partial goose observations via LLM merging."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model
        self._combined_map: str | None = None

    def update(self, goose_id: str, raw_observation: str) -> None:
        """Merge a new goose observation into the combined map."""
        if self._combined_map is None:
            prompt = raw_observation
        else:
            prompt = MAP_COMPOSER_MERGE_PROMPT.format(self._combined_map, goose_id, raw_observation)
        self._combined_map = get_model_answer(
            self._client, self._model, MAP_COMPOSER_SYSTEM_PROMPT, prompt
        )

    def get(self) -> str | None:
        """Return the current combined map, or None if no observations yet."""
        return self._combined_map


def get_model_answer(client, model, system_prompt, user_prompt):
    while True:
        try:
            model_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            content = model_response.choices[0].message.content if model_response.choices else None
            if not content:
                print("Retrying: empty response, sleeping 5s.")
                time.sleep(5)
                continue
            return content
        except openai.RateLimitError:
            print("Retrying: rate limit hit, sleeping 60s.")
            time.sleep(60)
        except Exception:
            traceback.print_exc()
            raise

class GooseAgentImpl(GooseAgent):
    def __init__(self, client: OpenAI, used_model: str, env: GooseEnvironment, append_to_chat: ChatCallback) -> None:
        super().__init__(client, used_model, env, append_to_chat)
        self._append_to_chat(f"Initialized {env.goose_id}.")
        self.turn_counter = 0
        self.map_composer: MapComposer | None = None  # set by planner after construction

    def on_call(self, message: GooseAgentMessage) -> GooseAgentResult:
        self._append_to_chat(f"Planner message: {message.description}")
        #event = self._env.honk(count=1)

        HOLD_KEYWORDS = ("honk", "wait", "stay", "hold", "remain", "stand", "do nothing")
        last_move = "HONK"
        if any(k in message.description.lower() for k in HOLD_KEYWORDS):
            self._env.honk(1)
        else:
            for attempt in range(3):
                answer = get_model_answer(self._client,
                                          self._used_model,
                                          GOOSE_SYSTEM_PROMPT.format(self._env.goose_id),
                                          GOOSE_PROMPT.format(self._env.describe_state(), message.description))
                self._append_to_chat("Goose move: " + answer)
                last = answer.strip().split('\n')[-1].strip().lower()
                if last == "up":
                    self._env.move(Direction.UP)
                    last_move = "UP"
                    break
                elif last == "down":
                    self._env.move(Direction.DOWN)
                    last_move = "DOWN"
                    break
                elif last == "left":
                    self._env.move(Direction.LEFT)
                    last_move = "LEFT"
                    break
                elif last == "right":
                    self._env.move(Direction.RIGHT)
                    last_move = "RIGHT"
                    break
                elif last == "honk":
                    self._env.honk(1)
                    last_move = "HONK"
                    break
                elif attempt == 2:
                    raise RuntimeError("Bad Gemma")

        if self.map_composer is not None:
            self.map_composer.update(self._env.goose_id, self._env.describe_state())

        goose_report = get_model_answer(self._client,
                                        self._used_model,
                                        GOOSE_SYSTEM_PROMPT.format(self._env.goose_id),
                                        GOOSE_OBS_PROMPT.format(self._env.describe_state(), last_move, OTHER_GOOSE[self._env.goose_id]))

        self._append_to_chat(f"Turn {self.turn_counter}. GooseAgent answer: {goose_report}")
        self.turn_counter += 1
        return GooseAgentResult(output=goose_report)


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
        self._map_composer = MapComposer(client, used_model)
        self._goal_estimator = GoalEstimator(client, used_model)
        for goose in self._agents.values():
            goose.map_composer = self._map_composer
        self._memory = []
        self._last_reports: dict[str, str] = {
            gid: "No reports yet." for gid in self._agents
        }
        self._prev_estimates: dict[str, str] = {
            gid: "UNKNOWN" for gid in self._agents
        }
        self.turn_counter = 0
        self._append_to_chat(f"Initialized planner for level: {env.level_name}.")
        self._planner_notes = "No planner message yet."

    def step(self) -> None:
        self._append_to_chat("Planner step executed.")
        combined_map = self._map_composer.get() or "No map yet."
        self._append_to_chat(f"Combined map:\n{combined_map}")
        positions = {
            gid: self._agents[gid]._env.visible_goose_positions().get(gid)
            for gid in sorted(self._agents)
        }
        estimates = {
            gid: self._goal_estimator.estimate(gid, combined_map, positions[gid])
            for gid in sorted(self._agents)
        }
        self._append_to_chat(f"Goal estimates: {estimates}")
        expanded_map = expand_map(combined_map)

        g1 = positions["goose_1"] if positions["goose_1"] is not None else (-1, -1)
        g2 = positions["goose_2"] if positions["goose_2"] is not None else (-1, -1)

        think_1 = get_model_answer(self._client, self._used_model,
                                   PLANNER_SYSTEM_PROMPT,
                                   PLANNER_THINK_PROMPT_GOOSE_1.format(
                                       self._planner_notes,
                                       expanded_map,
                                       self._env.task_description,
                                       self._last_reports["goose_1"],
                                       g1[0], g1[1],
                                       estimates["goose_1"],
                                       self._prev_estimates["goose_1"]))
        self._planner_notes = think_1.strip() or self._planner_notes

        think_2 = get_model_answer(self._client, self._used_model,
                                   PLANNER_SYSTEM_PROMPT,
                                   PLANNER_THINK_PROMPT_GOOSE_2.format(
                                       self._planner_notes,
                                       expanded_map,
                                       self._env.task_description,
                                       self._last_reports["goose_2"],
                                       f"x={g2[0]}, y={g2[1]}",
                                       estimates["goose_2"],
                                       self._prev_estimates["goose_2"]))
        self._planner_notes = think_2.strip() or self._planner_notes
        self._append_to_chat(f"Turn {self.turn_counter}. Planner notes: {self._planner_notes}")

        for goose_id, goose in sorted(self._agents.items()):
            pos = self._agents[goose_id]._env.visible_goose_positions().get(goose_id)
            row, col = pos if pos is not None else (-1, -1)
            if goose_id == "goose_1":
                user_prompt = PLANNER_PROMPT_GOOSE_1.format(
                    row, col, expanded_map,
                    estimates["goose_1"], estimates["goose_2"])
            else:
                user_prompt = PLANNER_PROMPT_GOOSE_2.format(
                    row, col, expanded_map,
                    estimates["goose_1"], estimates["goose_2"],
                    goose_id)
            answer = get_model_answer(self._client, self._used_model,
                                      PLANNER_SYSTEM_PROMPT, user_prompt)
            lines = [l for l in answer.strip().splitlines() if l.strip()]
            instruction = lines[-1] if lines else answer.strip()
            self._append_to_chat(f"Planner -> {goose_id}: {instruction}")

            result = goose.on_call(GooseAgentMessage(description=instruction))
            if result.error is not None:
                self._append_to_chat(f"{goose_id} error: {result.error}")
            else:
                expanded = expand_report(result.output)
                self._append_to_chat(f"{goose_id} result: {expanded}")
                self._memory.append((goose_id, expanded))
                self._last_reports[goose_id] = expanded
        self._prev_estimates = estimates
        self.turn_counter += 1