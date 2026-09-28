"""Validate conventional run names against the explicit branch algorithm.

This checks provenance only: names never select an implementation.
"""

def check_experiment_identity(*, task: str, experiment_name: str,
                              algorithm: str, policy_id: str) -> None:
    prefix = task + "-"
    if experiment_name.startswith(prefix):
        label, separator, run_id = experiment_name[len(prefix):].rpartition("-")
        if separator and run_id.isdecimal() and label != algorithm:
            raise ValueError(
                f"Experiment {experiment_name!r} is labeled {label!r}, but this "
                f"learner runs {algorithm!r} ({policy_id}). Select the matching "
                f"algorithm branch or use {task}-{algorithm}-{run_id}. "
                "The experiment name does not select the algorithm."
            )
    print(
        f"[ALGORITHM] {algorithm} policy={policy_id} "
        f"task={task} experiment={experiment_name}",
        flush=True,
    )
