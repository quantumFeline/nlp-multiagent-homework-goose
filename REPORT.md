Repository: https://github.com/quantumFeline/nlp-multiagent-homework-goose

# Goose game

## General design

Our task is to coordinate the geese. Gemma is not very good at spatial thinking with limited information, so we have to be mindful about how the responsibilities are split between the planner and the geese.

The overall approach in this implementation is the following: the planner tries to issue **high-level** tasks (find and press the button, go to the goal), but the **immediate** reasoning (how to move towards the button? how to go through that door?) is delegated to the goose.

This solution makes use of the new API which separates *system prompt* and *user prompt*. The system prompt is used for general information (such as map legend), while the user prompt describes the current game state.

This solution also makes use of **planner memory** - recent goose reports and a running game status note created by the planner itself and included into the next planner call. This ensures consistency in long-term task execution.

In hard mode, each goose only observes a partial view of the map. To reconstruct a fuller picture, we introduce a **map composer**: after each goose move, the goose's raw observation is passed to a separate LLM call that merges it into a running combined map, replacing unknown cells (`?`) with newly observed values while preserving previously known static elements. The combined map is passed to the planner on every turn, allowing it to reason about the full layout as it is gradually revealed.

## Problems encountered

### Misreading the coordinates

The goose reads the absolute coordinates and misinterprets them, assuming (x, y) format.

**Solution:**

* Emphasize the (row, col) format in the system prompts - or encourage to drop the format altogether.
* Request the planner to provide relative rather than absolute goals.

### Confusing the geese

Planner frequently gets confused which goose it is currently commanding.

**Solution:**

* Include the goose id in the system prompt.

### Forgetting the goal

Sometimes the planner sees the door is closed and commands a goose to stay on the button, even if the other goose has already passed through the door and there is no longer any need to do that.

**Solution:**

* Provide a detailed description of the game and the game rules. Describe the general mechanics (buttons open doors) without any level-specific guidance.
* Let the planner fill out its own memory to keep track of current tasks.
* Include example memory entries that handle door-button interactions.

### Wandering off

The geese frequently wander off the button or the goal, even if the planner requests them to stay.

**Solution:**

* Explicitly mention in system and user prompts for the goose that HONK command can be used for waiting or staying in place, and in fact should be used for that exact purpose.
* Short-circuit the LLM move call entirely when the planner instruction contains a hold keyword (honk, wait, stay, hold, remain); execute the honk directly without asking the model.

### Misreporting immediate surroundings

Free-form natural language goose observations (e.g. "there is a button to the north-east") introduced spatial errors - the model would describe diagonal positions incorrectly or conflate adjacency with distance. An earlier attempt at a structured format ("Immediately north: X. Also visible: ...") still produced errors in the "Also visible" free-form section, where the model described non-adjacent cells with incorrect relative directions.

**Solution:**

* Drop free-form descriptions entirely. Replace with a tightly structured single-line format covering only the four directly adjacent cells: `standing_on=[goal/square] North=[symbol] South=[symbol] West=[symbol] East=[symbol]`. This limits the model's task to reading four adjacent symbols, which is reliable at this scale.
* Apply **chain-of-thought prompting** across all LLM calls: the model is instructed to reason step by step, then end its response with a single line in a fixed format. Only the last line is parsed as the final answer, leaving reasoning free while keeping the extractable output unambiguous.

### Planner not reacting to state changes

When a door opened, the planner would issue instructions before processing the new GoalEstimator signal, causing the pressing goose to walk away just as the other was about to pass through.

**Solution:**

* Split planning into two sequential calls: a **think** call (`PLANNER_THINK_PROMPT`) that updates the notes given the latest estimates, followed by per-goose **instruction** calls that read the freshly updated notes.

### Planner confusing closed doors with buttons

The planner repeatedly identified `$` as a button and sent geese to press it. Removing the combined map from planner prompts eliminated the confusion but left the planner with no spatial knowledge; geese wandered aimlessly.

**Solution:**

* Keep the combined map but pre-process it before passing to the planner: `@` to `B`, `$` to `D`, `/` to `O`. Characters are distinct enough that the model does not conflate them.
* Expand raw symbols in goose reports to unambiguous names (`West=closed_door`, `South=button`) before storing them in planner memory.

### Self-location errors in goose calls

The goose would scan the map incorrectly to locate its own marker, leading to wrong adjacency reports and wrong move decisions.

**Solution:**

* Inject the goose's grid position from `visible_goose_positions()` as an anchor into the move prompt, observation prompt, and GoalEstimator prompt. The model is told its position explicitly and only needs to read four adjacent symbols.

## Technical problems encountered

* Gemma has a strict per-minute rate limit, resulting in `RateLimitError`. **Solution**: introduce a helper function that catches the exception and retries after a cooldown; empty responses are retried with a short sleep.
* Logs were hard to keep track of and compare across agents. **Solution**: introduce turn counters and log the combined map and planner notes on every step.
