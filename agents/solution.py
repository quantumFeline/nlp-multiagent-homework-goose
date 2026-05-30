import time
import traceback

import openai
from openai import OpenAI

from agents.base import ChatCallback, GooseAgent, GooseAgentMessage, GooseAgentResult, PlannerAgent
from goose_game.environment import GooseEnvironment, PlannerEnvironment
from goose_game.models import Direction

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
                         # "Pro tip: A goose standing on a button cannot see it: its own marker hides the button, so it will report standing on an ordinary square. "
                         # "Never wait for a goose to confirm it is on a button. Infer it from your own instructions: if you sent a goose toward a button "
                         # "and its next report no longer shows that button in an adjacent square, assume it is now standing on the button and command it to "
                         # "\"Stay on the button\" every turn until the other goose reports it has reached the goal.\n"
                         # "A held button only keeps a door open while the goose remains on it, so the holding goose must Stay (honk) continuously; "
                         # "do not move it away to pursue the goal until the other goose is confirmed at the goal.\n\n"
                         # "Pro tip: treat progress as latching. Once you have established that a goose is holding a button, "
                         # "or that a goose has passed through a door, record it and do not reverse it on the strength of a single later report unless that report plainly contradicts it. "
                         # "Commit to one sub-goal per goose; do not alternate a goose between two goals on consecutive turns.\n")

PLANNER_PROMPT = ("Previous goose reports were the following:\n"
                  "{}\n"
                  "Previous planner message was the following:\n"
                  "{}\n"
                  "The goal of this level is the following:\n"
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
                        "Here is how to read the game map legend:\n"
                        "# wall\n"
                        ". empty square\n"
                        "* goal (also shown when YOU are standing on it)\n"
                        "@ button\n"
                        "`$` closed door\n"
                        "`/` open door\n"
                        "`?` unknown\n"
                        "`X` goose 1\n"
                        "`Y` goose 2\n"
                            "The coordinates must be read as (row, col), i.e. (y, x).\n"
                            "Reason about positions only relative to yourself - compass directions and distances in squares. "
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
                        "Here is how to read the game map legend:\n"
                        "# wall\n"
                        ". empty square\n"
                        "* goal (also shown when YOU are standing on it)\n"
                        "@ button\n"
                        "`$` closed door\n"
                        "`/` open door\n"
                        "`?` unknown\n"
                        "`X` goose 1\n"
                        "`Y` goose 2\n"
                           "The coordinates must be read as (row, col), i.e. (y, x).\n")
GOOSE_OBS_PROMPT = ("The current game state that you see is the following:\n"
                    "{}\n"
                    "Find your own marker in the map (X if you are goose_1, Y if you are goose_2); "
                    "that square is where you are standing. Describe everything else relative to it - "
                    "compass direction and how many squares away. Do not use numeric coordinates.\n"
                    "Pass the key information to the planner in this format:\n"
                    "\"I am standing on [the goal / an ordinary square]. Immediately north: [symbol]. "
                    "Immediately south: [symbol]. Immediately east: [symbol]. Immediately west: [symbol]. "
                    "Also visible: [relative description].\"\n"
                    "Example descriptions:\n"
                    "\"I am standing on an ordinary square. Immediately north: empty. Immediately south: empty. "
                    "Immediately east: wall. Immediately west: empty. Also visible: a button two squares to the "
                    "north-east; the path south toward the goal looks clear.\"\n"
                    "\"I am standing on an ordinary square. Immediately north: open door. Immediately south: empty. "
                    "Immediately east: empty. Immediately west: goose_2. Also visible: the goal lies just north of "
                    "the open door.\"\n"
                    "\"I am standing on the goal. Immediately north: empty. Immediately south: wall. "
                    "Immediately east: empty. Immediately west: wall. Also visible: none.\"\n")

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

        # positions = self._env.visible_goose_positions()
        # self_state = self._env.describe_state()
        # result = f"Visible positions: {positions}, self-state: {self_state}"
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
                                                                             self._env.task_description))
            self._append_to_chat(f"Turn {self.turn_counter}. Planner notes: " + self._planner_notes)
        self.turn_counter += 1