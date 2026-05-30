import time
import traceback

import openai
from openai import OpenAI

from agents.base import ChatCallback, GooseAgent, GooseAgentMessage, GooseAgentResult, PlannerAgent
from goose_game.environment import GooseEnvironment, PlannerEnvironment
from goose_game.models import Direction

PLANNER_SYSTEM_PROMPT = ("You are a PLANNER in a GOOSE GAME. "
                         "Your task so to coordinate the geese so that both of them reach the goal marked with * and honk when standing on it.\n"
                         "Each game turn, a goose is asking you where to move. Your task is to provide it with its next action.\n"
                         "Once a goose reports reaching the goal, instruct it to HONK and stay put.\n"
                         "Sometimes geese may need help from each other, for example, by opening doors for them. "
                         "In this case, one goose should help the other, provided there is a closed door between the other goose and its next destination. "
                         "Once the other goose has passed through the door, the first one should no longer hold a button and instead "
                         "focus on pursuing the goal themselves.\n"
                         "You may also meet more complex situations, such as multiple buttons or multiple doors.\n")

PLANNER_PROMPT = ("Previous goose reports were the following:\n"
                  "{}\n"
                  "Previous planner message was the following:\n"
                  "{}\n"
                  "The goal of this level is the following:\n"
                  "{}\n"
                  "The following is the description of the game state as visible by GOOSE {}.\n"
                  "{}\n"
                  "Please choose the next goal for GOOSE {},"
                  "and describe it in a single sentence. Example valid commands:\n"
                      "\"Press the button approximately north of you.\"\n"
                      "\"Find and reach the goal in the southeast corner.\"\n"
                      "\"Go through an open door west of you.\"\n"
                      "\"Honk and wait.\"\n")

PLANNER_OBS_PROMPT = ("Previous goose reports were the following:\n"
                  "{}\n"
                  "Previous planner message was the following:\n"
                  "{}\n"
                  "The goal of this level is the following:\n"
                  "{}\n"
                  "Return the updated planner message that keeps the key game state information. Example messages:\n"
                    "\"I need to keep goose_1 on the button so that goose_2 can pass through the door.\"\n"
                    "\"I need to test which button can open the door for goose_1. The button at (3,2) didn't work, "
                      "therefore, the correct button is either at (3,4) or (5,5).\"\n"
                    "\"goose_2 has successfully passed the door. Now goose_1 may navigate to the goal.")

GOOSE_SYSTEM_PROMPT = ("You are GOOSE {} in a GOOSE game. Your task is to obey the PLANNER commands.\n"
                       "Based on the planner message, you need to figure out how to move towards the indicated position "
                       "(or how to stay in place if you are already where you should be).\n"
                       "The commands are high-level, and may require multiple moves. You need to figure out your next move only.\n"
                       "The move may be UP (north), DOWN (south), LEFT (west), RIGHT (east), or HONK (stay).\n"
                       "HONK is the move you should use if you want to do nothing. It is analogous to \"stay\", \"wait\", or \"hold\".\n"
                       "It is also the command both geese need to execute to successfully finish the game.\n")
GOOSE_PROMPT = ("The game state you are currently observing is the following:\n"
                "{}\n"
                "The map legend:\n"
                "# wall\n"
                ". empty square\n"
                "* goal (also shown when YOU are standing on it)\n"
                "@ button\n"
                "`$` closed door\n"
                "`/` open door\n"
                "`?` unknown\n"
                "`X` goose 1\n"
                "`Y` goose 2\n"
                "You receive a message from the PLANNER that tells you that your next goal is the following:\n"
                "{}\n"
                "If the map shows * at your current position (i.e., you cannot see your own X/Y symbol), you are standing on the goal. "
                "Your correct action is to HONK. You also need to HONK if you are told to stay in place.\n"
                "Otherwise, please choose your next turn: UP, DOWN, LEFT, RIGHT, or HONK. Print your next move only.\n")

GOOSE_OBS_SYSTEM_PROMPT = "You are GOOSE {} in a GOOSE game. You perceive and pass useful information to the planner.\n"
GOOSE_OBS_PROMPT = ("The current game state that you see is the following:\n"
                    "{}"
                    "and the visible goose positions are:\n"
                    "{}"
                    "The map legend:\n"
                    "# wall\n"
                    ". empty square\n"
                    "* goal (also shown when YOU are standing on it)\n"
                    "@ button\n"
                    "`$` closed door\n"
                    "`/` open door\n"
                    "`?` unknown\n"
                    "`X` goose 1\n"
                    "`Y` goose 2\n"
                    "Please pass the key information about your observations to the planner in an accessible form.\n"
                    "Example descriptions:\n"
                    "\"There is a wall to the east of me and a button northeast.\"\n"
                    "\"There is an open door directly north and another goose west.\"\n"
                    "\"There is a wall west of me and a wall south of me. There is the goal east of me.\n")

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

    def on_call(self, message: GooseAgentMessage) -> GooseAgentResult:
        self._append_to_chat(f"Planner message: {message.description}")
        #event = self._env.honk(count=1)

        if "honk" in message.description.lower() or "wait" in message.description.lower():
            self._append_to_chat("Honk!")
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
                                        GOOSE_OBS_PROMPT.format(self._env.describe_state(), self._env.visible_goose_positions()))
        #goose_report = f"My position is {self._env.visible_goose_positions()[self._env.goose_id]}. {goose_report}"

        self._append_to_chat(f"GooseAgent answer: {goose_report}")
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
        self._append_to_chat(f"Initialized planner for level: {env.level_name}.")

    def step(self) -> None:
        self._append_to_chat("Planner step executed.")

        result = GooseAgentResult(output="No reports yet.")
        planner_notes = "No planner message yet."
        for goose_id, goose in sorted(self._agents.items()):

            # Planning
            answer = get_model_answer(self._client,
                                      self._used_model,
                                      PLANNER_SYSTEM_PROMPT,
                                      PLANNER_PROMPT.format(self._memory[-4:],
                                          planner_notes,
                                          self._env.task_description,
                                                            goose_id,
                                                            result.output,
                                                            goose_id))

            task = GooseAgentMessage(description=answer) #f"{goose_id}, honk now."
            self._append_to_chat(f"Calling {goose_id}.")
            result = goose.on_call(task)
            if result.error is not None:
                self._append_to_chat(f"{goose_id} error: {result.error}")
            else:
                self._append_to_chat(f"{goose_id} result: {result.output}")
                self._memory.append((goose_id, result.output))

            # Planner self-notes
            planner_notes = get_model_answer(self._client,
                                      self._used_model,
                                      PLANNER_SYSTEM_PROMPT,
                                      PLANNER_OBS_PROMPT.format(self._memory[-4:],
                                          planner_notes,
                                          self._env.task_description))
            self._append_to_chat("Planner notes: " + planner_notes)
