"""Things you run by hand, and helpers that are not part of the training loop.

    stats.py           scan the train split once -> stats.json (bounds, mean, std)
    runs.py            table of every run in outputs/, sortable and filterable
    aggregate_seeds.py mean +- std across a seed sweep

The split against dataset/ is by WHEN the code runs, not by what it touches:
dataset/ is the per-batch hot path, utils/ is offline analysis that runs once and
writes a file. Keeping a dataset scan out of setup() is the difference between a
30-second startup on every run and a one-off command.
"""
