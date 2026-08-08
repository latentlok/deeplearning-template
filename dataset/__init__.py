"""Your data. Everything between "bytes on disk" and "a batch on the device".

    loader.py     ZarrData -- the one you rewrite. Reads a zarr group of named
                  variables from <data_root>/{train,val}.zarr.
    examples.py   the datamodules for the three shipped example models. Delete with them.

A DataModule owns four things: the Dataset, the split, the DataLoader arguments, and
the collate_fn that decides what a batch actually looks like. The Trainer never
inspects a batch -- it only moves it to the device -- so dicts, tuples, graph batches
and custom containers all work. Dicts are the convention here because they keep step
methods readable.

The data itself lives OUTSIDE the repo, at ${paths.data_root} (env var DL_DATA).
Statistics over that data are computed once, offline, by utils/stats.py -- never in
setup(). See loader.py for why.
"""
