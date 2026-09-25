import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def main():
    ap = argparse.ArgumentParser(description="Animate Brownian motion trajectories.")
    ap.add_argument("csv_path", help="CSV file produced by brownian.cpp")
    ap.add_argument("--trail", type=int, default=30,
                     help="number of past steps to show as a fading trail (default 30, 0 = dots only)")
    ap.add_argument("--particles", type=int, default=None,
                     help="only animate the first N particles (default: all)")
    ap.add_argument("--interval", type=int, default=30,
                     help="milliseconds between animation frames (default 30)")
    ap.add_argument("--save", type=str, default=None,
                     help="save animation to this file (e.g. out.mp4 or out.gif) instead of showing it live")
    ap.add_argument("--fps", type=int, default=30, help="frames per second when saving (default 30)")
    args = ap.parse_args()

    print(f"Loading {args.csv_path} ...")
    df = pd.read_csv(args.csv_path)

    if args.particles is not None:
        df = df[df.particle_id < args.particles]

    steps = sorted(df.step.unique())
    particle_ids = sorted(df.particle_id.unique())
    n_particles = len(particle_ids)
    n_steps = len(steps)
    print(f"{n_particles} particles, {n_steps} logged frames")

    pivot_x = df.pivot(index="step", columns="particle_id", values="x").reindex(steps)
    pivot_y = df.pivot(index="step", columns="particle_id", values="y").reindex(steps)
    X = pivot_x.values
    Y = pivot_y.values

    has_temp = "temperature" in df.columns
    if has_temp:
        temp_per_step = df.groupby("step")["temperature"].first().reindex(steps).values
    else:
        temp_per_step = None

    time_per_step = df.groupby("step")["time"].first().reindex(steps).values if "time" in df.columns else steps

    x_min, x_max = np.nanmin(X), np.nanmax(X)
    y_min, y_max = np.nanmin(Y), np.nanmax(Y)
    pad_x = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    pad_y = 0.05 * (y_max - y_min if y_max > y_min else 1.0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")

    colors = plt.cm.viridis(np.linspace(0, 1, n_particles))

    scat = ax.scatter(X[0], Y[0], s=15, c=colors)
    trail_lines = []
    if args.trail > 0:
        for i in range(n_particles):
            (line,) = ax.plot([], [], lw=0.8, alpha=0.5, color=colors[i])
            trail_lines.append(line)

    title = ax.set_title("")

    def title_text(frame_idx):
        t = time_per_step[frame_idx]
        if has_temp:
            return f"step {steps[frame_idx]}  |  t = {t:.4g} s  |  T = {temp_per_step[frame_idx]:.1f} K"
        return f"step {steps[frame_idx]}  |  t = {t:.4g} s"

    def update(frame_idx):
        scat.set_offsets(np.column_stack([X[frame_idx], Y[frame_idx]]))
        if args.trail > 0:
            start = max(0, frame_idx - args.trail)
            for i, line in enumerate(trail_lines):
                line.set_data(X[start:frame_idx + 1, i], Y[start:frame_idx + 1, i])
        title.set_text(title_text(frame_idx))
        return [scat, title, *trail_lines]

    anim = animation.FuncAnimation(
        fig, update, frames=n_steps, interval=args.interval, blit=False
    )

    if args.save:
        print(f"Saving animation to {args.save} (this may take a while) ...")
        anim.save(args.save, fps=args.fps)
        print("Done.")
    else:
        plt.show()


if __name__ == "__main__":
    main()
