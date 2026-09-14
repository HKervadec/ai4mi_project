import unittest

from src.config import config_hash, load_config, merge, parse_overrides, read_yaml

CFG = "configs/segthor_enet_ce.yaml"


class ConfigTests(unittest.TestCase):
    def test_experiment_overrides_base(self):
        cfg = load_config(CFG)
        self.assertEqual(cfg["experiment"], "segthor_enet_ce")
        self.assertEqual(cfg["model"]["kwargs"], {"kernels": 8, "factor": 2})
        self.assertEqual(cfg["optim"]["kwargs"]["lr"], 0.0005)  # inherited from base

    def test_unknown_key_rejected_but_kwargs_free_form(self):
        with self.assertRaises(KeyError):
            merge({"train": {"epochs": 1}}, {"train": {"epoch": 2}})
        merged = merge({"model": {"kwargs": {}}}, {"model": {"kwargs": {"depth": 4}}})
        self.assertEqual(merged["model"]["kwargs"]["depth"], 4)

    def test_cli_overrides_are_typed(self):
        self.assertEqual(parse_overrides(["train.epochs=5", "optim.kwargs.lr=1e-3", "wandb.tags=[a, b]"]),
                         {"train": {"epochs": 5}, "optim": {"kwargs": {"lr": 1e-3}}, "wandb": {"tags": ["a", "b"]}})
        self.assertEqual(load_config(CFG, ["train.epochs=3"])["train"]["epochs"], 3)
        # PyYAML alone reads `1e-4` as the string '1e-4' (YAML 1.1); config files must get a float
        self.assertEqual(read_yaml("lr: 1e-4\nb: [0.9, 1.5E+2]\nname: e5"), {"lr": 1e-4, "b": [0.9, 150.0], "name": "e5"})

    def test_smoke_layer_then_overrides(self):
        cfg = load_config(CFG, ["wandb.mode=offline"], smoke=True)
        self.assertEqual(cfg["train"]["epochs"], 2)
        self.assertEqual(cfg["wandb"]["mode"], "offline")
        self.assertTrue(cfg["paths"]["run_root"].endswith("_smoke"))

    def test_hash_ignores_notes_and_wandb(self):
        a = load_config(CFG)
        self.assertEqual(config_hash(a), config_hash(load_config(CFG, ["notes=hi", "wandb.mode=disabled"])))
        self.assertNotEqual(config_hash(a), config_hash(load_config(CFG, ["seed=1"])))


if __name__ == "__main__":
    unittest.main()
