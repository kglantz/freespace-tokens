# Local data

This directory is **git-ignored** (except this file). Never commit dataset files.

## Expected layout

Place (or symlink) the ScanNet++ dataset at `data/scannetpp/`:

```bash
ln -s /path/to/your/scannetpp data/scannetpp
```

Alternatively, keep the data anywhere and set `dataset.root` in `configs/local.yaml`.

The exact internal directory structure (scene folders, iPhone vs. DSLR modality,
file names, depth units) will be documented here after inspecting the downloaded
data in Milestone 1.

- Dataset website: https://scannetpp.mlsg.cit.tum.de/scannetpp/
- Official toolkit: https://github.com/scannetpp/scannetpp
