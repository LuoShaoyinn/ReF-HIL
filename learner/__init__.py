"""Single branch-local IQL-floor plus action-limited SAC implementation.

The package intentionally performs no eager imports, so the actor can load
`learner.policy` without constructing learner transport, replay, or logging
dependencies.
"""
