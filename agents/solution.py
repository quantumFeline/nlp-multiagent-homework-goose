import time

from openai import OpenAI

from agents.base import ChatCallback, GooseAgent, GooseAgentMessage, GooseAgentResult, PlannerAgent
from goose_game.environment import GooseEnvironment, PlannerEnvironment
from goose_game.models import Direction

PLANNER_PROMPT_START = ("You are a PLANNER in a GOOSE GAME. Your task so to coordinate the geese so that all of them reach X and honk when standing on X."
                "Each game turn, a goose is asking you where to move. Your task is to provide it with its next action."
                "The following is the description of the game state as visible by one of the geese."
                "The parts of the labyrinth the goose is unaware of are marked with X.\n")
PLANNER_PROMPT_END = "Please choose your GOOSE's next turn: UP, DOWN, LEFT, RIGHT, or HONK. Print your next move only.\n"

GOOSE_PROMPT_START = ("You are a GOOSE in a GOOSE game. Your task is to obey the PLANNER commands."
                      "You receive a message from the PLANNER that tells you the following:")
GOOSE_PROMPT_END = "Please choose your next turn: UP, DOWN, LEFT, RIGHT, or HONK. Print your next move only.\n"

def get_model_answer(client, model, prompt):
    answer = None
    while answer is None:
        model_response = client.responses.create(
            model=model,
            input=prompt
        )

        if model_response.error is not None:
            if model_response.error.code == 429:  # rate limit exceeded
                time.sleep(60)
            else:
                raise Exception(model_response.error.code)
        else:
            answer = model_response.output[0].content[0].text
    return answer

class GooseAgentImpl(GooseAgent):
    def __init__(self, client: OpenAI, used_model: str, env: GooseEnvironment, append_to_chat: ChatCallback) -> None:
        super().__init__(client, used_model, env, append_to_chat)
        self._append_to_chat(f"Initialized {env.goose_id}.")

    def on_call(self, message: GooseAgentMessage) -> GooseAgentResult:
        self._append_to_chat(f"Planner message: {message.description}")
        #event = self._env.honk(count=1)

        answer = get_model_answer(self._client,
                                  self._used_model,
                                  PLANNER_PROMPT_START + message.description + PLANNER_PROMPT_END)

        if answer == "UP":
            self._env.move(Direction.UP)
        elif answer == "DOWN":
            self._env.move(Direction.DOWN)
        elif answer == "LEFT":
            self._env.move(Direction.LEFT)
        elif answer == "RIGHT":
            self._env.move(Direction.RIGHT)
        elif answer == "HONK":
            self._env.honk(1)
        else:
            raise RuntimeError("Bad Gemma")

        positions = self._env.visible_goose_positions()
        self_state = self._env.describe_state()
        result = f"Visible positions: {positions}, self-state: {self_state}"

        self._append_to_chat(f"GooseAgent answer: {result}")
        return GooseAgentResult(output=result)


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
        self._append_to_chat(f"Initialized planner for level: {env.level_name}.")

    def step(self) -> None:
        self._append_to_chat("Planner step executed.")

        result = GooseAgentResult(output="[THE GAME IS NOT INITIALIZED YET. PLEASE HONK POLITELY]")
        for goose_id, goose in sorted(self._agents.items()):

            answer = get_model_answer(self._client,
                                      self._used_model,
                                      PLANNER_PROMPT_START + result.output + PLANNER_PROMPT_END)

            task = GooseAgentMessage(description=answer)#f"{goose_id}, honk now.")
            self._append_to_chat(f"Calling {goose_id}.")
            result = goose.on_call(task)
            if result.error is not None:
                self._append_to_chat(f"{goose_id} error: {result.error}")
            else:
                self._append_to_chat(f"{goose_id} result: {result.output}")
