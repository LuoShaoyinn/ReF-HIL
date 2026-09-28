import unittest

from shared.experiment_identity import check_experiment_identity


class ExperimentIdentityTest(unittest.TestCase):
    def test_hgdagger_cannot_start_with_hilserl_name(self):
        with self.assertRaisesRegex(ValueError, "does not select the algorithm"):
            check_experiment_identity(task="insert_gear",
                experiment_name="insert_gear-hilserl-0",
                algorithm="hgdagger", policy_id="hgdagger_gaussian_bc_v1")

    def test_hilserl_cannot_start_with_hgdagger_name(self):
        with self.assertRaises(ValueError):
            check_experiment_identity(task="insert_gear",
                experiment_name="insert_gear-hgdagger-0",
                algorithm="hilserl", policy_id="hilserl_scalar_sac_v1")

    def test_matching_or_unstructured_names_allowed(self):
        for name in ("insert_gear-hilserl-1", "unnamed", "debug"):
            check_experiment_identity(task="insert_gear", experiment_name=name,
                algorithm="hilserl", policy_id="hilserl_scalar_sac_v1")
