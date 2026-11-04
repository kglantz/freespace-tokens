# Tests

Install the CPU dependencies and run the tests from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

Tests cover camera conventions, depth decoding, TSDF fusion, visibility, state assignment,
and exact cuboid partitioning. The ScanNet++ integration tests use local data when available.

For the real-scene diagnostics, see the [three-state map](../documents/stage_one_validation.md)
and [cuboid validation](../documents/cuboid_validation.md) reports.
