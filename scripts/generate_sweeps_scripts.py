"""
Sweep definitions and input-file generation for wacdiff scaling runs.

These are the sweeps we are interested in:

strong scaling, nodes
    nodes(2^{0..6}), cpus(8), size(512)
strong scaling, cpus
    nodes(8), cpus(2^{0..6}), size(512)
weak scaling, nodes
    n=0..6: nodes(2^n), cpus(8), size(round(256 * 2^{n/3}))
weak scaling, cpus
    n=0..6: nodes(8), cpus(2^n), size(round(256 * 2^{n/3}))
io overhead
    n=0..6: nodes(2^n), cpus(8), size(512), checkpoint(32)
"""

from dataclasses import dataclass

# Constants we may be interested in tuning.
REPETITIONS = 3
MAX_ITERATIONS = 1_024        # 2_048 gives >= ~30s at the largest scale
SCALE = 0.1
STRONG_SIZE = 512             # strong-scaling global grid (per dimension)
WEAK_BASE_SIZE = 256          # weak-scaling base grid (per dimension)
IO_FREQUENCY = 32             # checkpoint interval for the I/O scaling sweep

@dataclass
class Run:
    nodes: int
    cpus_per_node: int
    size: int
    io_frequency: int = 0


@dataclass
class Sweeps:
    strong_scaling_node: list[Run]
    strong_scaling_cpu: list[Run]
    weak_scaling_node: list[Run]
    weak_scaling_cpu: list[Run]
    io_overhead: list[Run]


def generate_sweeps() -> Sweeps:
    sweeps = Sweeps([], [], [], [], [])
    for n in range(0, 6 + 1):
        weak_size = round(WEAK_BASE_SIZE * 2 ** (n / 3))
        sweeps.strong_scaling_node.append(Run(2**n, 8, STRONG_SIZE))
        sweeps.strong_scaling_cpu.append(Run(8, 2**n, STRONG_SIZE))
        sweeps.weak_scaling_node.append(Run(2**n, 8, weak_size))
        sweeps.weak_scaling_cpu.append(Run(8, 2**n, weak_size))
        sweeps.io_overhead.append(Run(2**n, 8, STRONG_SIZE, IO_FREQUENCY))
    return sweeps


def generate_input(run: Run) -> str:
    input_string = f"""<?xml version="1.0" encoding="UTF-8"?>
<simulation>

    <!-- Iterations between saved checkpoints (0 disables I/O) -->
    <checkpoint>{run.io_frequency}</checkpoint>

    <!-- Thermal diffusivity constant α -->
    <diffusion>0.05</diffusion>

    <discretization>
        <time_step>0.01</time_step>
        <cell_size>{SCALE}</cell_size>
        <dimensions>
            <dim0>{run.size}</dim0>
            <dim1>{run.size}</dim1>
            <dim2>{run.size}</dim2>
        </dimensions>
    </discretization>

    <!-- type="constant": single value; type="expr": f(x, y, z) expression -->
    <initial_conditions type="expr">
        if ((x^2 + y^2 + z^2)^(1/2) &lt; {run.size}/2 * {SCALE}) 0; else 100;
    </initial_conditions>

    <boundary_conditions>
        <axis0 type="periodic"/>
        <axis1 type="periodic"/>
        <axis2 type="periodic"/>
    </boundary_conditions>

    <convergence_conditions>
        <iterations_under>{MAX_ITERATIONS}</iterations_under>
        <avg_delta_over>0.0</avg_delta_over> <!-- run exactly iterations_under steps -->
    </convergence_conditions>

</simulation>"""
    return input_string


def main():
    pass

if __name__ == "__main__":
    main()
