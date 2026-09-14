"""Reference agents.

Every agent module exposes the same entry point used by `evaluate.py`:

    def run(env: RaceEnv, tools: list[StructuredTool], attempts: int = 1, **kwargs) -> None

* dummy   - drives straight at half throttle (crashes; shows the loop mechanics)
* reflex  - rule-based organiser sanity check: proves the track is drivable
* hermes  - the zero-shot LLM baseline students must beat
"""
