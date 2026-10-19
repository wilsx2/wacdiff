# wacdiff

A parallel, distributed 3D heat diffusion simulator using the explicit Finite Difference Method (FDM). Designed for HPC workloads, it combines MPI for distributed-memory parallelism across nodes with OpenMP for shared-memory threading within each node.

## Features

- **3D explicit FDM solver** for the heat equation on a structured Cartesian grid
- **Hybrid MPI + OpenMP parallelism**. Cartesian domain decomposition with persistent halo exchanges
- **Parallel HDF5 output**. All ranks write collectively to a single `.h5` file
- **XDMF metadata**. Output is directly consumable by ParaView and VisIt
- **Flexible configuration** via XML input files

## Building

Prerequisites:
- CMake >= 3.5
- C++ compiler with C++26 support
- MPI implementation
- OpenMP (Bundled with GCC/Clang)
- Parallel HDF5

```bash
git clone https://github.com/wilsx2/wacdiff.git
cd wacdiff

# Release build
cmake -S . -B build/release
cmake --build build/release

# Debug/profiling build
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug
```

## Running

```bash
# Single process
./build/release/wacheat input.xml

# Distributed across 8 MPI ranks
mpirun -n 8 ./build/release/wacheat input.xml

# Custom output directory and log level
mpirun -n 8 ./build/release/wacheat input.xml --output ./results --log info
```

### CLI Options

| Argument | Required | Default | Description |
|---|---|---|---|
| `input` | Yes | N/A | Path to XML input file |
| `--output` / `-o` | No | `.` | Directory for output files |
| `--log` | No | `err` | Log level: `trace`, `debug`, `info`, `warn`, `err`, `critical`, `off` |

## Configuration

Simulations are fully defined by an XML file. Example:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<simulation>

    <!-- Iterations between saved checkpoints -->
    <checkpoint>100</checkpoint>

    <!-- Thermal diffusivity constant α -->
    <diffusion>0.05</diffusion>

    <discretization>
        <time_step>0.01</time_step>
        <cell_size>0.1</cell_size>
        <dimensions>
            <dim0>256</dim0>
            <dim1>256</dim1>
            <dim2>256</dim2>
        </dimensions>
    </discretization>

    <!-- type="constant": single value; type="expr": f(x, y, z) expression -->
    <initial_conditions type="expr">
        if (x &lt; 256/2*0.1) 0; else 100;
    </initial_conditions>

    <boundary_conditions>
        <axis0 type="periodic"/>
        <axis1 type="periodic"/>
        <axis2 type="periodic"/>
    </boundary_conditions>

    <convergence_conditions>
        <iterations_under>10000</iterations_under>
        <avg_delta_over>0.00001</avg_delta_over>
    </convergence_conditions>

</simulation>
```

### Initial Conditions

| Type | Description |
|---|---|
| `constant` | Fills the entire domain with a single numeric value |
| `expr` | Evaluates an ExprTk expression at each cell's physical coordinates `x`, `y`, `z` |

### Boundary Conditions (per axis)

| Type | Description |
|---|---|
| `periodic` | Wraps the axis. opposite faces are connected |
| `dirichlet` | Fixed temperature value on the face |
| `neumann` | Fixed flux (gradient) on the face |

Non-periodic axes specify `first` and `last` faces independently:

```xml
<axis2>
    <first type="dirichlet">0.0</first>
    <last  type="neumann">0.0</last>
</axis2>
```

## Output

Each run produces two files in the output directory:

| File | Description |
|---|---|
| `sim.h5` | Parallel HDF5 file; dataset `/N` holds the full 3D temperature array at iteration N |
| `sim.xmf` | XDMF XML metadata linking the HDF5 data to mesh geometry |

The XDMF file can be opened directly in **ParaView** or **VisIt** to visualize the time series.
Note: When in ParaView, apply the Point Data to Cell Data filter.

## License

GNU General Public License v3.0. See [LICENSE.txt](LICENSE.txt).
