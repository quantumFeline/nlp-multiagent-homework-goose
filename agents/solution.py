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
                         "You may also meet more complex situations, such as multiple buttons or multiple doors. Always keep track of what has been achieved so far.\n\n")

PLANNER_PROMPT = ("Previous goose reports were the following:\n"
                  "{}\n"
                  "Previous planner message was the following:\n"
                  "{}\n"
                  "The goal of this level is the following:\n"
                  "{}\n"
                  "Combined map built from all goose observations so far (? = not yet seen):\n"
                  "{}\n"
                  "The following is the description of the game state as visible by GOOSE {}.\n"
                  "{}\n"
                  "The coordinates must be read as (row, col), i.e. (y, x).\n"
                  "Please choose the next goal for GOOSE {},"
                  "and describe it in a single sentence. "
                  "Do not use absolute coordinates, rather relying on relative descriptions. Example valid commands:\n"
                      "\"Press the button north of you.\"\n"
                      "\"Stay on the button.\"\n"
                      "\"Find and reach the goal.\"\n"
                      "\"Go through an open door west of you.\"\n"
                      "\"Find and reach the closed door.\"\n"
                      "\"Honk and wait.\"\n\n"
                  "Do not use complex commands, such as \"Go through the door and then reach the goal.\" Only provide the next course of action.")

PLANNER_OBS_PROMPT = ("Previous goose reports were the following:\n"
                  "{}\n"
                  "Previous planner message was the following:\n"
                  "{}\n"
                  "The goal of this level is the following:\n"
                  "{}\n"
                  "Combined map built from all goose observations so far (? = not yet seen):\n"
                  "{}\n"
                  "Return the updated planner message that keeps the key game state information. Example messages:\n"
                    "\"goose_2 is separate from the goal by a door. I need to keep goose_1 on the button at (5, 0) so that goose_2 can pass through the door at (1, 1).\"\n"
                    "\"goose_1 is separate from the goal by a door. I need to test which button can open the door for goose_1. The button at (3,2) didn't work, "
                      "therefore, the correct button is either at (3,4) or (5,5). I will now lead goose_1 to the button at (3,4), which is north of it.\"\n"
                    "\"goose_2 has successfully passed the door. goose_1 must now ignore the button and navigate directly to the goal.\n")

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
                       + "Reason about positions only relative to yourself (compass directions and distances in squares). "
                         "Do not use numeric coordinates.\n")
GOOSE_PROMPT = ("The game state you are currently observing is the following:\n"
                "{}\n"
                "You receive a message from the PLANNER that tells you that your next goal is the following:\n"
                "{}\n"
                "If the map shows * at your current position (i.e., you cannot see your own X/Y symbol), you are standing on the goal. "
                "Your correct action is to HONK.\n"
                "HONK is also the move you should use if you want to do nothing. It is analogous to \"stay\", \"wait\", or \"hold\".\n"
                "Otherwise, please choose your next turn: UP, DOWN, LEFT, RIGHT, or HONK. Print your next move only.\n")

GOOSE_OBS_SYSTEM_PROMPT = ("You are GOOSE {} in a GOOSE game. You perceive and pass useful information to the planner.\n"
                           "Map legend: " + MAP_LEGEND
                           + "Note: * is also shown when YOU are standing on the goal (your own marker is hidden).\n")
GOOSE_OBS_PROMPT = ("The current game state that you see is the following:\n"
                    "{}\n"
                    "Find your own marker (X if you are goose_1, Y if you are goose_2). "
                    "Report the symbol in each of the four cells directly adjacent to you, "
                    "and whether you are standing on the goal or an ordinary square.\n"
                    "Use this exact format (one line):\n"
                    "standing_on=[goal/square] North=[symbol] South=[symbol] West=[symbol] East=[symbol]\n"
                    "Example: standing_on=square North=. South=@ West=# East=.\n"
                    "Example: standing_on=goal North=. South=# West=. East=.\n")

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
    answer = None
    while answer is None:
        # model_response = client.responses.create(
        #     model=model,
        #     input=prompt
        # )
        try:
            model_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            if not model_response.choices:
                continue # retry
            answer = model_response.choices[0].message.content
        except openai.RateLimitError:
            time.sleep(60)
        except Exception:
            traceback.print_exc()
            raise
    return answer

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
                answer = get_model_answer(self._client,
                                          self._used_model,
                                          GOOSE_SYSTEM_PROMPT.format(self._env.goose_id),
                                          GOOSE_PROMPT.format(self._env.describe_state(), message.description))
                self._append_to_chat("Goose move: " + answer)
                answer = answer.lower()
                if "up" in answer:
                    self._env.move(Direction.UP)
                    break
                elif "down" in answer:
                    self._env.move(Direction.DOWN)
                    break
                elif "left" in answer:
                    self._env.move(Direction.LEFT)
                    break
                elif "right" in answer:
                    self._env.move(Direction.RIGHT)
                    break
                elif "honk" in answer:
                    self._env.honk(1)
                    break
                elif attempt == 2:
                    raise RuntimeError("Bad Gemma")

        if self.map_composer is not None:
            self.map_composer.update(self._env.goose_id, self._env.describe_state())

        goose_report = get_model_answer(self._client,
                                        self._used_model,
                                        GOOSE_OBS_SYSTEM_PROMPT.format(self._env.goose_id),
                                        GOOSE_OBS_PROMPT.format(self._env.describe_state()))
        #goose_report = f"My position is {self._env.visible_goose_positions()[self._env.goose_id]}. {goose_report}"

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
        for goose_id, goose in sorted(self._agents.items()):
            answer = get_model_answer(self._client,
                                      self._used_model,
                                      PLANNER_SYSTEM_PROMPT,
                                      PLANNER_PROMPT.format(self._memory[-4:],
                                                            self._planner_notes,
                                                            self._env.task_description,
                                                            self._map_composer.get() or "No map yet.",
                                                            goose_id,
                                                            self._last_reports[goose_id],
                                                            goose_id))
            task = GooseAgentMessage(description=answer)
            self._append_to_chat(f"Calling {goose_id}.")
            result = goose.on_call(task)
            if result.error is not None:
                self._append_to_chat(f"{goose_id} error: {result.error}")
            else:
                self._append_to_chat(f"{goose_id} result: {result.output}")
                self._memory.append((goose_id, result.output))
                self._last_reports[goose_id] = result.output

            self._planner_notes = get_model_answer(self._client,
                                                   self._used_model,
                                                   PLANNER_SYSTEM_PROMPT,
                                                   PLANNER_OBS_PROMPT.format(self._memory[-4:],
                                                                             self._planner_notes,
                                                                             self._env.task_description,
                                                                             self._map_composer.get() or "No map yet."))
            self._append_to_chat(f"Turn {self.turn_counter}. Planner notes: " + self._planner_notes)
        self.turn_counter += 1