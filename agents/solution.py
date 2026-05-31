import time
import traceback

import openai
from openai import OpenAI

from agents.base import ChatCallback, GooseAgent, GooseAgentMessage, GooseAgentResult, PlannerAgent
from goose_game.environment import GooseEnvironment, PlannerEnvironment
from goose_game.models import Direction

MAP_LEGEND = "# wall  . empty  * goal  @ button  $ closed door  / open door  ? unknown  X goose_1  Y goose_2\n"

PLANNER_SYSTEM_PROMPT = ("You are a PLANNER in a GOOSE GAME.\n"
                         "The game is a puzzle where both geese need to reach the goal. However, they might have to solve intermediate tasks in order to achieve that.\n"
                         "Your task so to coordinate the geese so that both of them reach the goal marked with * and honk when standing on it.\n"
                         "Each game turn, a goose is provides you with what it sees and asks you where to move. Your task is to provide it with its next action.\n"
                         "Remember that a goose may only have partial information about the game state and does not have memory. You have memory and need to "
                         "use it for long-term planning.\n"
                         "Once a goose reports reaching the goal, instruct it to HONK and stay put.\n\n"
                         "The game has the following types of objects:\n"
                         "goose_1 and goose_2 that you control;\n"
                         "goal - a special square that both geese need to reach;\n"
                         "doors - they are impassable when closed, possible to go through when open; a button opens a door;\n"
                         "buttons - they can be pressed by standing on them, however, not every button is connected to a door, and some of them might do nothing;\n"
                         "walls - they are impassable.\n\n"
                         "Pro tip: sometimes geese may need help from each other, for example, by opening doors for each other. "
                         "In this case, one goose should help the other, provided there is a closed door between the other goose and its next destination. "
                         "Once the other goose has passed through the door, the first one should no longer hold a button and instead "
                         "focus on pursuing the goal themselves.\n"
                         "Remember that a goose cannot hold the door for itself.\n"
                         "You may also meet more complex situations, such as multiple buttons or multiple doors. Always keep track of what has been achieved so far.\n"
                         "If a goose is blocked by a closed door, instruct it to HONK and wait. Send the other goose to press buttons one by one until the door opens, tracking which buttons have been tried.\n\n")

PLANNER_THINK_PROMPT = (
    "Previous goose reports:\n{}\n"
    "Previous planner notes:\n{}\n"
    "Level goal: {}\n"
    "Combined map (? = not yet seen):\n{}\n"
    "Goal reachability: goose_1 is {}, goose_2 is {}.\n"
    "(CLEAR = unobstructed path; BLOCKED = closed door on every path; UNKNOWN = unobserved cells blocking view)\n"
    "If a goose just became CLEAR that was previously BLOCKED, the door opened - update the plan accordingly.\n"
    "Think step by step, then end with a single line:\n"
    "NOTES: <one sentence summarising current game state and plan for both geese>\n"
    "Example endings:\n"
    "NOTES: goose_1 is holding the button at (5,0) but the door has not opened; will try the button at (3,0) next.\n"
    "NOTES: door opened - both geese are now CLEAR and should head directly to the goal.\n"
    "NOTES: goose_2 has passed through the door; goose_1 must leave the button and navigate to the goal.\n"
)

PLANNER_PROMPT = (
    "Planner notes:\n{}\n"
    "Level goal: {}\n"
    "Game state as visible by GOOSE {}:\n{}\n"
    "Based on the notes above, choose the single next action for GOOSE {}.\n"
    "One sentence, relative directions only, no coordinates.\n"
    "Example instructions: Press the button north of you. / Stay on the button. / "
    "Find and reach the goal. / Honk and wait. / Go through the open door to the west.\n"
    "End with: INSTRUCTION: <your instruction>\n"
)

GOOSE_SYSTEM_PROMPT = ("You are GOOSE {} in a GOOSE game. Your task is to obey the PLANNER commands.\n"
                       "Based on the planner message, you need to figure out how to move towards the indicated position "
                       "(or how to stay in place if you are already where you should be).\n"
                       "The commands are high-level, and may require multiple moves. You need to figure out your next move only.\n"
                       "The move may be UP (north), DOWN (south), LEFT (west), RIGHT (east), or HONK (stay).\n"
                       "HONK is the move you should use if you want to do nothing. It is analogous to \"stay\", \"wait\", or \"hold\".\n"
                       "It is also the command both geese need to execute to successfully finish the game.\n"
                       "Map legend: " + MAP_LEGEND
                       + "Note: * is also shown when YOU are standing on the goal (your own marker is hidden).\n"
                       + "The coordinates must be read as (row, col), i.e. (y, x).\n"
                       + "Cells beyond the map boundary are walls (#).\n"
                       + "Reason about positions only relative to yourself (compass directions and distances in squares). "
                         "Do not use numeric coordinates.\n")
GOOSE_PROMPT = ("The game state you are currently observing is the following:\n"
                "{}\n"
                "You are at row {}, col {}.\n"
                "You receive a message from the PLANNER that tells you that your next goal is the following:\n"
                "{}\n"
                "If the map shows * at your current position (i.e., you cannot see your own X/Y symbol), you are standing on the goal. "
                "Your correct action is to HONK.\n"
                "HONK is also the move you should use if you want to do nothing. It is analogous to \"stay\", \"wait\", or \"hold\".\n"
                "Think step by step about your position, the planner's goal, and what move brings you closer to it. "
                "End your response with a single line containing only your final move: UP, DOWN, LEFT, RIGHT, or HONK.\n")

GOOSE_OBS_SYSTEM_PROMPT = ("You are GOOSE {} in a GOOSE game. You perceive and pass useful information to the planner.\n"
                           "Map legend: " + MAP_LEGEND
                           + "Note: * is also shown when YOU are standing on the goal (your own marker is hidden).\n"
                           + "Cells beyond the map boundary are walls (#). Treat them as # without further comment.\n")
GOOSE_OBS_PROMPT = ("The current game state that you see is the following:\n"
                    "{}\n"
                    "You are at row {}, col {}. "
                    "Read the symbol in each directly adjacent cell.\n"
                    "End your response with a single line in this exact format:\n"
                    "standing_on=[symbol] North=[symbol] South=[symbol] West=[symbol] East=[symbol]\n"
                    "Example last line: standing_on=. North=. South=@ West=# East=.\n"
                    "Example last line: standing_on=* North=. South=# West=. East=.\n")

MAP_COMPOSER_SYSTEM_PROMPT = (
    "You are a map merger for a grid-based game. "
    "You receive a combined map built so far and a new partial observation, and output an updated combined map.\n"
    "Map symbols: " + MAP_LEGEND
    + "Merging rules:\n"
    "- ? means the cell was not yet observed. Replace ? with the new observation's value if it is known there.\n"
    "- Static elements (#, ., *, @, $, /) once known never revert to ?. Keep them from the combined map even "
    "if the new observation shows ? there.\n"
    "- Goose positions (X, Y) are dynamic. Always take them from the new observation, not the old combined map.\n"
    "- If a cell is known in both maps and they disagree on a static element, trust the new observation "
    "(doors can open or close).\n"
    "Output ONLY the updated grid, row by row, with no extra text, explanation, or blank lines.\n"
)

MAP_COMPOSER_FIRST_PROMPT = (
    "This is the first observation. Output it as-is as the initial combined map.\n"
    "Observation:\n{}\n"
    "Output ONLY the grid."
)

MAP_COMPOSER_MERGE_PROMPT = (
    "Current combined map:\n{}\n\n"
    "New observation from {}:\n{}\n\n"
    "Output the updated combined map. ONLY the grid, no other text."
)


GOAL_ESTIMATOR_SYSTEM_PROMPT = (
    "You are analyzing a game map to determine if a goose can reach the goal.\n"
    "Map symbols: " + MAP_LEGEND +
    "A goose is CLEAR if there is at least one path to the goal with no closed doors ($) or walls (#).\n"
    "A goose is BLOCKED if there is a closed door ($) or wall (#) on every path between it and the goal (*).\n"
    "A goose is UNKNOWN if it is not CLEAR and there are unobserved cells (?) between it and the goal that may or may not contain obstacles.\n"
    "Think step by step, then end your response with exactly one word on its own line: CLEAR, BLOCKED, or UNKNOWN.\n"
)

GOAL_ESTIMATOR_PROMPT = (
    "Combined map:\n{}\n"
    "{} is at row {}, col {}.\n"
    "Can {} reach the goal (*)?\n"
    "Think step by step, then end with CLEAR, BLOCKED, or UNKNOWN."
)


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
            prompt = MAP_COMPOSER_FIRST_PROMPT.format(raw_observation)
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
        if any(k in message.description.lower() for k in HOLD_KEYWORDS):
            self._env.honk(1)
        else:
            for attempt in range(3):
                pos = self._env.visible_goose_positions().get(self._env.goose_id, (-1, -1))
                answer = get_model_answer(self._client,
                                          self._used_model,
                                          GOOSE_SYSTEM_PROMPT.format(self._env.goose_id),
                                          GOOSE_PROMPT.format(self._env.describe_state(), pos[0], pos[1], message.description))
                self._append_to_chat("Goose move: " + answer)
                last = answer.strip().split('\n')[-1].strip().lower()
                if last == "up":
                    self._env.move(Direction.UP)
                    break
                elif last == "down":
                    self._env.move(Direction.DOWN)
                    break
                elif last == "left":
                    self._env.move(Direction.LEFT)
                    break
                elif last == "right":
                    self._env.move(Direction.RIGHT)
                    break
                elif last == "honk":
                    self._env.honk(1)
                    break
                elif attempt == 2:
                    raise RuntimeError("Bad Gemma")

        if self.map_composer is not None:
            self.map_composer.update(self._env.goose_id, self._env.describe_state())

        pos = self._env.visible_goose_positions().get(self._env.goose_id, (-1, -1))
        goose_report = get_model_answer(self._client,
                                        self._used_model,
                                        GOOSE_OBS_SYSTEM_PROMPT.format(self._env.goose_id),
                                        GOOSE_OBS_PROMPT.format(self._env.describe_state(), pos[0], pos[1]))

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
        self.turn_counter = 0
        self._append_to_chat(f"Initialized planner for level: {env.level_name}.")
        self._planner_notes = "No planner message yet."

    def step(self) -> None:
        self._append_to_chat("Planner step executed.")
        combined_map = self._map_composer.get() or "No map yet."
        self._append_to_chat(f"Combined map:\n{combined_map}")
        estimates = {
            gid: self._goal_estimator.estimate(
                gid,
                combined_map,
                self._agents[gid]._env.visible_goose_positions().get(gid),
            )
            for gid in sorted(self._agents)
        }
        self._append_to_chat(f"Goal estimates: {estimates}")

        think = get_model_answer(self._client, self._used_model,
                                 PLANNER_SYSTEM_PROMPT,
                                 PLANNER_THINK_PROMPT.format(self._memory[-4:],
                                                             self._planner_notes,
                                                             self._env.task_description,
                                                             combined_map,
                                                             estimates["goose_1"],
                                                             estimates["goose_2"]))
        notes_line = next((l.removeprefix("NOTES:").strip() for l in think.splitlines() if l.startswith("NOTES:")), self._planner_notes)
        self._planner_notes = notes_line
        self._append_to_chat(f"Turn {self.turn_counter}. Planner notes: {self._planner_notes}")

        for goose_id, goose in sorted(self._agents.items()):
            answer = get_model_answer(self._client,
                                      self._used_model,
                                      PLANNER_SYSTEM_PROMPT,
                                      PLANNER_PROMPT.format(self._planner_notes,
                                                            self._env.task_description,
                                                            goose_id,
                                                            self._last_reports[goose_id],
                                                            goose_id))
            instruction = next((l.removeprefix("INSTRUCTION:").strip() for l in answer.splitlines() if l.startswith("INSTRUCTION:")), answer.strip().splitlines()[-1])
            self._append_to_chat(f"Planner -> {goose_id}: {instruction}")

            result = goose.on_call(GooseAgentMessage(description=instruction))
            if result.error is not None:
                self._append_to_chat(f"{goose_id} error: {result.error}")
            else:
                self._append_to_chat(f"{goose_id} result: {result.output}")
                self._memory.append((goose_id, result.output))
                self._last_reports[goose_id] = result.output
        self.turn_counter += 1