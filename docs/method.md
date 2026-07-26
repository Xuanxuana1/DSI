# DSI data contract

The preprocessing pipeline converts each impression into a typed watch outcome,
a discretized watch-time bin, model context, and evaluation metadata. The four
outcomes are early skip, mid watch, completion, and overplay. Dataset-specific
file layouts are defined in `configs/datasets/`.
