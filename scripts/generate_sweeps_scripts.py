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
io scaling
    n=0..6: nodes(2^n), cpus(8), size(512), checkpoint(32)
"""

import os
from dataclasses import dataclass

# Constants we may be interested in tuning.
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
    io_scaling: list[Run]

def generate_name(run: Run):
    return f"wacdiff-N{run.nodes}-C{run.cpus_per_node}-S{run.size}-IO{run.io_frequency}"

def generate_sweeps() -> Sweeps:
    sweeps = Sweeps([], [], [], [], [])
    for n in range(0, 6 + 1):
        weak_size = round(WEAK_BASE_SIZE * 2 ** (n / 3))
        sweeps.strong_scaling_node.append(Run(2**n, 8, STRONG_SIZE))
        sweeps.strong_scaling_cpu.append(Run(8, 2**n, STRONG_SIZE))
        sweeps.weak_scaling_node.append(Run(2**n, 8, weak_size))
        sweeps.weak_scaling_cpu.append(Run(8, 2**n, weak_size))
        sweeps.io_scaling.append(Run(2**n, 8, STRONG_SIZE, IO_FREQUENCY))
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
        ((x - {run.size / 2 * SCALE})^2 + (y - {run.size / 2 * SCALE})^2 + (z - {run.size / 2 * SCALE})^2)^(1/2)
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

def generate_batch_script(run: Run) -> str:
    name = generate_name(run)
    script_contents = f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --output=slurm-{name}-%j.out
#SBATCH --error=slurm-{name}-%j.err
#SBATCH --time=
#SBATCH --ntasks={run.nodes}
#SBATCH --cpus-per-task={run.cpus_per_node}

module load gcc/11.3.0
module load openmpi/gcc/11.3.0/zen2/
module load hdf5/gcc/11.3.0/openmpi/4.1.5/zen2/

srun build/wacdiff inputs/{name}.xml -o outputs/{name}
"""
    return script_contents

def write_sweep_files(sweeps: Sweeps):
    sweep_attrs = [
        "strong_scaling_node",
        "strong_scaling_cpu",
        "weak_scaling_node",
        "weak_scaling_cpu",
        "io_scaling"
    ]

    os.makedirs(f"inputs", exist_ok=True)
    os.makedirs(f"outputs", exist_ok=True)
    os.makedirs(f"sbatch", exist_ok=True)
    for attr in sweep_attrs:
        runs = getattr(sweeps, attr)
        for run in runs:
            name = generate_name(run)
            with open(f"inputs/{name}.xml", "w", encoding="utf-8") as file:
                content = generate_input(run)
                file.write(content)
            with open(f"sbatch/{name}.sh", "w", encoding="utf-8") as file:
                content = generate_batch_script(run)
                file.write(content)

def main():
    sweeps = generate_sweeps()
    write_sweep_files(sweeps)

if __name__ == "__main__":
    main()
