import pathlib, yaml

for src, dst, seeds in [
    ("experiments/collusion/configs/hackathon_2_1_a_pair_private_channel.yaml", "experiments/collusion/configs/batch_a_private_10seeds.yaml", list(range(1, 13))),
    ("experiments/collusion/configs/hackathon_2_1_b_pair_no_channel.yaml", "experiments/collusion/configs/batch_b_no_channel_10seeds.yaml", list(range(1, 13))),
]:
    cfg = yaml.safe_load(pathlib.Path(src).read_text(encoding="utf-8"))
    cfg["experiment"]["max_concurrent_runs"] = 5
    cfg["experiment"]["seeds"] = seeds
    pathlib.Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print("Wrote", dst)
