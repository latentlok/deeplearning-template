"""Your models. One file per model, each a TaskModule.

A model file holds the architecture, the loss and the three required methods --
training_step, validation_step, configure_optimizers. It does NOT hold the data;
that is dataset/, because the usual shape of a project is one dataset and many
models tried against it.

Add a model:
  1. models/yours.py            subclass TaskModule
  2. configs/model/yours.yaml   _target_: models.yours.YourModel
  3. one line in MODEL_DATA in tests/test_contracts.py   -> contract tests for free

Shipped, all deletable:
  mlp.py       two-layer net; the fast smoke test, and the file to copy
  forecast.py  windowed timeseries; teacher forcing, scalers, free-running rollout
  pinn.py      du/dx = -u with a gradient loss and adaptive term weighting
"""
