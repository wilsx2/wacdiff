# Node- and core-hour calculation for the wacdiff sweeps

This document explains how the node-hour and core-hour estimates for the
sweeps in `scripts/generate_sweeps_scripts.py` were produced.

## 1. Baseline measurement

The baseline is a run of the existing `input.xml` on this machine:

| Quantity | Value |
|---|---|
| Grid | 64 × 64 × 64 = 262,144 cells |
| Iterations | 10,000 |
| Cell-updates | 64³ × 10,000 = 2.62144 × 10⁹ |
| CPUs used | 8 (single node) |
| Wall time | 22 s |

A compute-only run (checkpoint disabled) measured ~20.6 s, so the 22 s figure
is conservative.

Derived quantities:

```
baseline node-hours  = 22 s / 3600 = 0.006111 node-hours
baseline throughput  = 2.62144e9 / 22 ≈ 119.2 million cell-updates/s per 8-CPU node
```

## 2. Sweep definitions

The script now defines these constants:

| Parameter | Value |
|---|---|
| `REPETITIONS` | 3 |
| `MAX_ITERATIONS` | 1,024 |
| `STRONG_SIZE` | 512 |
| `WEAK_BASE_SIZE` | 256 |
| `IO_FREQUENCY` | 32 |
| `avg_delta_over` | 0.0 (runs exactly `MAX_ITERATIONS` steps) |

Each `Run` carries an `io_frequency` (checkpoint interval). The scaling
sweeps use the default `io_frequency = 0` (I/O disabled). The I/O overhead
sweep uses a fixed interval of 32.

| Sweep | nodes | cpus/node | size | io_frequency |
|---|---|---|---|---|
| strong scaling, nodes | 1, 2, 4, 8, 16, 32, 64 | 8 | 512 | 0 |
| strong scaling, cpus | 8 | 1, 2, 4, 8, 16, 32, 64 | 512 | 0 |
| weak scaling, nodes | 2ⁿ, n = 0..6 | 8 | round(256 · 2^(n/3)) | 0 |
| weak scaling, cpus | 8 | 2ⁿ, n = 0..6 | round(256 · 2^(n/3)) | 0 |
| I/O overhead | 2ⁿ, n = 0..6 | 8 | 512 | 32 |

The weak-scaling sizes are rounded to integers: 256, 323, 406, 512, 645,
813, 1024.

The I/O overhead sweep is a single-interval sweep: checkpoint every 32
iterations, with node count varying from 1 to 64 (7 runs). It measures how
checkpoint cost scales with node count rather than write frequency.

## 3. Model and assumptions

The estimate assumes:

1. Runtime scales linearly with the number of cells and the number of
   iterations.
2. Throughput scales linearly with total cores
   (`nodes × cpus_per_node`).
3. Perfect scaling: no halo-exchange, MPI, OpenMP, or load-imbalance
   overhead.
4. Per-core performance is the same as the baseline machine.
5. I/O time is excluded from the compute estimate. This is exact for the
   scaling sweeps (`io_frequency = 0`) but makes the I/O overhead sweep a
   lower bound, since I/O is the quantity that sweep measures.

## 4. Formula

For a run with grid size `S`, iterations `I`, nodes `N`, and
`cpus_per_node = C`:

```
wall_seconds = 22 × (S / 64)³ × (I / 10000) × (8 / (N × C))

node_hours   = N × wall_seconds / 3600
             = 0.006111 × (S / 64)³ × (I / 10000) × (8 / C)

core_hours   = node_hours × C
             = 0.006111 × (S / 64)³ × (I / 10000) × 8
```

Note that `N` cancels out of the node-hour expression: under perfect
scaling, doubling the node count halves the wall time, so node-hours stay
constant. Core-hours are independent of both `N` and `C`, as expected for a
fixed amount of work.

## 5. Worked example

Take the first strong-scaling run (`nodes = 1`, `cpus = 8`, `S = 512`,
`I = 1024`):

```
node_hours = 0.006111 × (512 / 64)³ × (1024 / 10000) × (8 / 8)
           = 0.006111 × 8³ × 0.1024 × 1
           = 0.006111 × 52.4288
           = 0.3204 node-hours
```

## 6. Results

### strong scaling, nodes (size 512, cpus 8, 1024 iterations)

| nodes | node-hours |
|---:|---:|
| 1 | 0.320 |
| 2 | 0.320 |
| 4 | 0.320 |
| 8 | 0.320 |
| 16 | 0.320 |
| 32 | 0.320 |
| 64 | 0.320 |
| **subtotal** | **2.243** |

### strong scaling, cpus (8 nodes, size 512, 1024 iterations)

| cpus/node | node-hours |
|---:|---:|
| 1 | 2.563 |
| 2 | 1.282 |
| 4 | 0.641 |
| 8 | 0.320 |
| 16 | 0.160 |
| 32 | 0.080 |
| 64 | 0.040 |
| **subtotal** | **5.086** |

### weak scaling, nodes (cpus 8, 1024 iterations)

| nodes | size | node-hours |
|---:|---:|---:|
| 1 | 256 | 0.040 |
| 2 | 323 | 0.080 |
| 4 | 406 | 0.160 |
| 8 | 512 | 0.320 |
| 16 | 645 | 0.641 |
| 32 | 813 | 1.283 |
| 64 | 1024 | 2.563 |
| **subtotal** | | **5.087** |

### weak scaling, cpus (8 nodes, 1024 iterations)

| cpus/node | size | node-hours |
|---:|---:|---:|
| 1 | 256 | 0.320 |
| 2 | 323 | 0.322 |
| 4 | 406 | 0.320 |
| 8 | 512 | 0.320 |
| 16 | 645 | 0.320 |
| 32 | 813 | 0.321 |
| 64 | 1024 | 0.320 |
| **subtotal** | | **2.243** |

### I/O overhead (size 512, cpus 8, checkpoint 32, 1024 iterations)

| nodes | node-hours |
|---:|---:|
| 1 | 0.320 |
| 2 | 0.320 |
| 4 | 0.320 |
| 8 | 0.320 |
| 16 | 0.320 |
| 32 | 0.320 |
| 64 | 0.320 |
| **subtotal** | **2.243** |

The compute-only node-hours are identical for every run. Actual wall time
will be larger because each run also writes 33 checkpoints
(`1 + 1024 / 32`) of 1 GiB each:

```
data per run  = 33 × 1 GiB = 33 GiB
data per sweep = 7 × 33 GiB = 231 GiB (693 GiB with 3 repetitions)
```

## 7. Totals

Compute-only, single repetition:

| Sweep | Node-hours | Core-hours |
|---|---:|---:|
| strong scaling, nodes | 2.243 | 17.94 |
| strong scaling, cpus | 5.086 | 17.94 |
| weak scaling, nodes | 5.087 | 40.70 |
| weak scaling, cpus | 2.243 | 40.70 |
| I/O overhead | 2.243 | 17.94 |
| **Total** | **16.90** | **135.2** |

With `REPETITIONS = 3`:

| Quantity | Value |
|---|---:|
| Node-hours | **50.71** |
| Core-hours | **405.7** |

Core-hours are computed as `node_hours × cpus_per_node`.

## 8. Caveats

- **Perfect scaling is optimistic.** Real halo/MPI/OpenMP overhead and load
  imbalance will make runs slower, especially at the largest core counts.
- **I/O time is excluded** from these numbers. For the scaling sweeps that
  is correct (`io_frequency = 0`). For the I/O overhead sweep, the
  compute-only node-hours are a lower bound and the measured I/O time is the
  whole point of the sweep.
- **The I/O overhead sweep measures node-count scaling at one interval.**
  It does not sweep checkpoint frequency. If the write-frequency overhead
  curve is also needed, add a small interval sweep at a fixed node count
  (e.g. `checkpoint = 10, 50, 100, 200, 500`).
- **Per-core speed varies by machine.** The baseline is an 8-thread laptop
  (4 physical cores); a cluster node with more/faster cores will change the
  absolute numbers, but the scaling formula remains the same.
- The original sweep definitions (1024³ strong scaling, 100,000 iterations)
  would have been about **6,440 node-hours** under the same assumptions.
