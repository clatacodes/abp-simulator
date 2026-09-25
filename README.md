# Brownian Motion Particle Simulator (C++)

## Build
```bash
g++ -O2 -std=c++17 -o brownian brownian.cpp
```

## Run
```bash
./brownian --particles 500 --steps 5000 --temp 300 --out trajectory.csv
```

## Key options
| Flag              | Meaning                                  | Default   |
|--------------------|-------------------------------------------|-----------|
| `--particles N`    | number of particles                       | 200       |
| `--steps N`         | number of timesteps                        | 2000      |
| `--dt SECONDS`      | timestep size                              | 1e-4      |
| `--temp KELVIN`     | temperature (the adjustable variable)      | 300       |
| `--viscosity PA_S`   | fluid viscosity                            | 1e-3 (water) |
| `--radius METERS`    | particle radius                            | 1e-6 (1 micron) |
| `--seed N`           | RNG seed (reproducibility)                 | random    |
| `--out FILE`         | CSV trajectory output                      | trajectory.csv |
| `--log-every N`      | write every Nth step to CSV                | 1         |
| `--temp-schedule "t1:T1,t2:T2"` | change temperature mid-run at given step indices | none |

## Physics
- Diffusion coefficient from the Stokes-Einstein relation:
  `D = kB*T / (6*pi*eta*r)`
- Each step, x and y displacements are drawn from `N(0, 2*D*dt)`.
- Console output reports measured mean-squared-displacement (MSD) against
  the theoretical `4*D*t` (2D) prediction every ~10% of the run, so you can
  verify the simulation is physically correct as you tweak parameters.

## Output
CSV columns: `step,time,particle_id,x,y,temperature`

Plot it with anything — e.g. in Python:
```python
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("trajectory.csv")
for pid, g in df[df.particle_id < 10].groupby("particle_id"):
    plt.plot(g.x, g.y)
plt.show()
```

## Example: see temperature effect directly
```bash
./brownian --temp 300 --out low_temp.csv --seed 1
./brownian --temp 900 --out high_temp.csv --seed 1
```
Compare the MSD growth rate printed to console — it scales linearly with T.
