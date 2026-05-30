# Goose game

## General design

Our task is to coordinate the geese. Gemma is not very good at spatial thinking with limited information, so we have to be mindful about how the responsibilities are split between the planner and the geese.

The overall approach in this implementation is the following: the planner tries to issue **high-level** tasks (find and press the button, go to the goal), but the **immediate** reasoning (how to move towards the button? how to go through that door?) is delegate to the goose.

This solution makes use of the new API which separates *system prompt* and *user prompt*. The system prompt is used for general information (such as map legend), while the user prompt describes the current game state.

This solution also makes user of **memory** - recent commands as well as the current game status created by the planner itself and included into the next call of the planner loop. This ensures consistency in long-term task execution.

## Problems encountered

### Misreading the coordinates

The goose reads the absolute coordinates and misinterprets it, assuming (x, y) format.

**Solution:**

* Emphasize the (row, col) format in the system prompts.
* Request the planner to provide relative rather than absolute goals.

### Confusing the geese

Planner frequently gets confused which goose it is currently commanding.

**Solution:**

* Include the goose id in the system prompt.

### Forgetting the goal

Sometimes planner sees the door is closed and commands a goose to stay on the button, even if the other goose has already passed through the door and there is no longer any need in doing that.

**Solution:**

* Let the planner fill out its own memory to keep track of current tasks.
* Include example memory commands that handle door-button interactions.

### Wandering off

The geese frequently wander off the button or the goal, even if the planner requests them to stay.

**Solution:**

* Explicitly mention in system and user prompts for the goose that HONK command can be used for waiting or staying in place, and in fact should be used for that exact purpose.

## Technical problems encountered

* Gemma has somewhat strict per-minute limit, which gets spent almost immediately, resulting in `UsageLimitError`. **Solution**: introduce a helper function that can handle retries automatically.
* Logging as hard to keep track of and compare against each other. **Solution**: introduce turn counters.