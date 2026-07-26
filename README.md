# DSI

This repository contains the code used for the DSI experiments in the paper.
It includes dataset preprocessing, the typed and time-only distribution
providers, the shared decision readout, the main-table baselines, matched
readout controls, and frozen transfer probes. Raw datasets and trained
artifacts are not included.

## Layout

```text
dsi/
├── dsi/                    # provider, losses, metrics, and decision readout
├── baselines/              # main-table baselines and controlled probes
├── preprocess/             # KuaiRec, KuaiRand-1K, and WeChat21 adapters
├── configs/
│   ├── datasets/           # paths and dataset-specific protocol
│   └── experiments/        # paper-level training and readout settings
├── docs/                   # processed-data contract
├── tests/                  # self-contained synthetic smoke test
└── run.py                  # public command line
```

## Installation

```bash
cd dsi
python -m pip install -e .
```

The code requires Python 3.10 or later. CUDA is optional. The smoke test runs
on CPU.

## Data

Download each dataset from its official source and place it under `data/raw`:

```text
data/raw/
├── kuairec/
├── kuairand-1k/
└── wechat21/
```

The expected source filenames are documented by the corresponding file in
`configs/datasets/`. Paths may be changed in YAML without modifying code.

Preprocess one dataset with:

```bash
python run.py prepare --config configs/datasets/kuairec.yaml
python run.py prepare --config configs/datasets/kuairand_1k.yaml
python run.py prepare --config configs/datasets/wechat21.yaml
```

Each command writes `standard_train.npz`, `standard_val.npz`,
`standard_test.npz`, and `manifest.json` under `artifacts/processed`.
Generated manifests, resolved configurations, cache metadata, and experiment
reports store repository paths relatively and redact external absolute paths as
`<external>`.

## Main method

Train the typed distribution provider and the paper decision readout:

```bash
python run.py train --config configs/datasets/kuairec.yaml --seed 2027 --device cuda:0
python run.py readout --config configs/datasets/kuairec.yaml --seed 2027 --device cuda:0
```

Use seeds `2027`, `2028`, and `2029` for the reported three-seed protocol.
The main readout uses the full training split, DCN, bounded ordinal basis
integration for watch time, and an overplay utility branch. Providers are
trained on the full split. Matched-readout, interface-width, and time-only
controls cap readout training at one million impressions; boundary sensitivity
uses the full split for both provider and readout. These settings live in
`configs/experiments/paper.yaml`.

## Paper controls

The time-only provider control uses the same training entry:

```bash
python run.py train --config configs/datasets/kuairec.yaml --variant time-only --seed 2027 --device cuda:0
```

The representation-interface controls reuse the typed checkpoint:

```bash
python run.py readout --config configs/datasets/kuairec.yaml --interface scalar --seed 2027 --device cuda:0
python run.py readout --config configs/datasets/kuairec.yaml --interface compact --seed 2027 --device cuda:0
python run.py readout --config configs/datasets/kuairec.yaml --interface full --seed 2027 --device cuda:0
python run.py readout --config configs/datasets/kuairec.yaml --interface plain --seed 2027 --device cuda:0
```

The appendix mechanism controls have a single entry. `raw`, `posterior`, and
`value-only` reuse the main provider. `nll-only`, `time-only`, and
`plain-summary` train an isolated provider when required:

```bash
python run.py ablation --config configs/datasets/kuairec.yaml --variant value-only --seed 2027 --device cuda:0
python run.py ablation --config configs/datasets/kuairec.yaml --variant nll-only --seed 2027 --device cuda:0
python run.py ablation --config configs/datasets/kuairec.yaml --variant time-only --seed 2027 --device cuda:0
```

One event-boundary sensitivity point rebuilds labels and retrains the full
pipeline in an isolated artifact directory:

```bash
python run.py boundary --config configs/datasets/kuairec.yaml --boundary beta --value 1.05 --seed 2027 --device cuda:0
```

Supported main-table baselines are `vr`, `wlr`, `cwm`, `difl`, `tpm`,
`cread`, `egmn`, `d2q`, and `gr`:

```bash
python run.py baseline --config configs/datasets/kuairec.yaml --method d2q --seed 2027 --device cuda:0
```

Run the shared-readout control with D2Q, TPM, or EGMN:

```bash
python run.py matched --config configs/datasets/kuairec.yaml --method d2q --seed 2027 --device cuda:0
```

Run frozen transfer probes with `raw`, `random-dsi`, `dsi`, `d2q`, `tpm`,
`cread`, or `egmn` representations:

```bash
python run.py transfer --config configs/datasets/kuairec.yaml --source dsi --task watch --seed 2027 --device cuda:0
python run.py transfer --config configs/datasets/kuairand_1k.yaml --source dsi --task engagement --seed 2027 --device cuda:0
```

## Smoke test

The smoke test creates a temporary synthetic processed dataset. It trains both
providers, fits the shared readout, trains VR, and checks MAE, XAUC, Long-N@5,
and OP-N@5 output:

```bash
python run.py smoke
```

No synthetic or real data is retained after the test.
