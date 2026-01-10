"""
Visually striking figures showing fern/Purkinje similarity:
Both are low-D structures embedded in high-D spaces.

Fern: fractal as byproduct of low-D GRN construction
Purkinje: fractal-like geometry functionally tuned for delay sampling

Different mechanisms, same geometric pattern.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from scipy.integrate import odeint

np.random.seed(42)

# Color schemes
FERN_GREEN = '#228B22'
FERN_DARK = '#006400'
PURKINJE_PURPLE = '#8B008B'
PURKINJE_LIGHT = '#DDA0DD'
FIBER_COLORS = plt.cm.viridis(np.linspace(0.2, 0.9, 10))
DELAY_CMAP = plt.cm.plasma


def barnsley_fern(n_points=100000):
    """Generate Barnsley fern using iterated function system."""
    x, y = 0, 0
    points = []

    for _ in range(n_points):
        r = np.random.random()
        if r < 0.01:
            x_new = 0
            y_new = 0.16 * y
        elif r < 0.86:
            x_new = 0.85 * x + 0.04 * y
            y_new = -0.04 * x + 0.85 * y + 1.6
        elif r < 0.93:
            x_new = 0.2 * x - 0.26 * y
            y_new = 0.23 * x + 0.22 * y + 1.6
        else:
            x_new = -0.15 * x + 0.28 * y
            y_new = 0.26 * x + 0.24 * y + 0.44
        x, y = x_new, y_new
        points.append((x, y))

    return np.array(points)


def generate_purkinje_arbor(depth=8, angle_spread=35, length_decay=0.7):
    """Generate a Purkinje-like dendritic arbor using L-system."""
    segments = []

    def branch(x, y, angle, length, current_depth, delay):
        if current_depth == 0 or length < 0.5:
            return

        # Main branch
        x_end = x + length * np.cos(np.radians(angle))
        y_end = y + length * np.sin(np.radians(angle))
        segments.append(((x, y), (x_end, y_end), delay, current_depth))

        # Small random variation
        angle_var = np.random.uniform(-5, 5)
        length_var = np.random.uniform(0.9, 1.1)

        # Branch with delay increment (longer paths = more delay)
        new_delay = delay + length * 0.1

        # Left branch
        branch(x_end, y_end, angle + angle_spread + angle_var,
               length * length_decay * length_var, current_depth - 1, new_delay)
        # Right branch
        branch(x_end, y_end, angle - angle_spread + angle_var,
               length * length_decay * length_var, current_depth - 1, new_delay)

        # Sometimes add extra branches
        if np.random.random() < 0.3 and current_depth > 3:
            extra_angle = np.random.uniform(-20, 20)
            branch(x_end, y_end, angle + extra_angle,
                   length * 0.5, current_depth - 2, new_delay)

    branch(0, 0, 90, 15, depth, 0)
    return segments


def lorenz(state, t, sigma=10, rho=28, beta=8/3):
    """Lorenz system."""
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def create_main_figure():
    """Create the main comparative visualization."""

    fig = plt.figure(figsize=(16, 12), facecolor='black')

    # === Panel A: Barnsley Fern ===
    ax1 = fig.add_subplot(2, 3, 1, facecolor='black')
    fern = barnsley_fern(80000)

    # Color by height (y-coordinate) for depth effect
    colors = plt.cm.YlGn(np.clip(fern[:, 1] / 10, 0, 1))
    ax1.scatter(fern[:, 0], fern[:, 1], c=colors, s=0.1, alpha=0.7)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(0, 10.5)
    ax1.axis('off')
    ax1.set_title('(a) Barnsley Fern\nFractal light capture',
                  color='white', fontsize=12, pad=10)

    # === Panel B: Purkinje Arbor ===
    ax2 = fig.add_subplot(2, 3, 2, facecolor='black')
    segments = generate_purkinje_arbor(depth=9, angle_spread=32)

    # Extract delays for coloring
    max_delay = max(s[2] for s in segments)

    for (x1, y1), (x2, y2), delay, depth in segments:
        # Color by delay (purple gradient)
        color = DELAY_CMAP(delay / max_delay)
        linewidth = 0.3 + depth * 0.3
        ax2.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=0.8)

    # Add soma
    soma = Circle((0, -2), 3, color=PURKINJE_PURPLE, alpha=0.8)
    ax2.add_patch(soma)
    ax2.text(0, -2, 'Soma', ha='center', va='center', color='white', fontsize=8)

    ax2.set_xlim(-25, 25)
    ax2.set_ylim(-8, 55)
    ax2.axis('off')
    ax2.set_title('(b) Purkinje Dendritic Arbor\nColor = conduction delay',
                  color='white', fontsize=12, pad=10)

    # Add colorbar for delay
    sm = plt.cm.ScalarMappable(cmap=DELAY_CMAP, norm=plt.Normalize(0, max_delay))
    cbar = plt.colorbar(sm, ax=ax2, fraction=0.03, pad=0.02)
    cbar.set_label('Delay (a.u.)', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # === Panel C: Parallel Fiber Array ===
    ax3 = fig.add_subplot(2, 3, 3, facecolor='black')

    n_fibers = 30
    for i in range(n_fibers):
        y_pos = i * 2
        # Fiber length varies systematically
        length = 20 + i * 3
        delay = length / 30  # Delay proportional to length

        color = DELAY_CMAP(delay / 4)
        ax3.plot([0, length], [y_pos, y_pos], color=color, linewidth=1.5, alpha=0.8)

        # Add small markers at regular intervals (synaptic contacts)
        for x in np.arange(5, length, 8):
            ax3.plot(x, y_pos, 'o', color=color, markersize=2, alpha=0.6)

    # Add input arrow
    ax3.annotate('', xy=(0, 30), xytext=(-10, 30),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax3.text(-12, 30, 'Input\nx(t)', ha='right', va='center', color='white', fontsize=10)

    # Add output label
    ax3.text(110, 30, 'τ₁, τ₂, τ₃...\nDelay taps', ha='left', va='center',
             color='white', fontsize=10)

    ax3.set_xlim(-15, 120)
    ax3.set_ylim(-5, 65)
    ax3.axis('off')
    ax3.set_title('(c) Parallel Fiber Array\nSystematic delay gradient',
                  color='white', fontsize=12, pad=10)

    # === Panel D: Lorenz Attractor (original) ===
    ax4 = fig.add_subplot(2, 3, 4, projection='3d', facecolor='black')

    t = np.linspace(0, 50, 5000)
    traj = odeint(lorenz, [1, 1, 1], t)

    # Color by time
    points = traj.reshape(-1, 1, 3)
    segments_3d = np.concatenate([points[:-1], points[1:]], axis=1)

    colors = plt.cm.plasma(np.linspace(0, 1, len(t)-1))
    for i in range(len(segments_3d)):
        ax4.plot3D(segments_3d[i, :, 0], segments_3d[i, :, 1], segments_3d[i, :, 2],
                   color=colors[i], linewidth=0.5, alpha=0.8)

    ax4.set_facecolor('black')
    ax4.xaxis.pane.fill = False
    ax4.yaxis.pane.fill = False
    ax4.zaxis.pane.fill = False
    ax4.xaxis.pane.set_edgecolor('gray')
    ax4.yaxis.pane.set_edgecolor('gray')
    ax4.zaxis.pane.set_edgecolor('gray')
    ax4.tick_params(colors='white')
    ax4.set_xlabel('X', color='white')
    ax4.set_ylabel('Y', color='white')
    ax4.set_zlabel('Z', color='white')
    ax4.set_title('(d) Original High-D Dynamics\n(Lorenz attractor)',
                  color='white', fontsize=12, pad=10)

    # === Panel E: 1D Observation ===
    ax5 = fig.add_subplot(2, 3, 5, facecolor='black')

    signal = traj[:1000, 0]
    time = t[:1000]

    # Color segments by value
    points_2d = np.array([time, signal]).T.reshape(-1, 1, 2)
    segments_2d = np.concatenate([points_2d[:-1], points_2d[1:]], axis=1)

    norm = plt.Normalize(signal.min(), signal.max())
    lc = LineCollection(segments_2d, cmap='plasma', norm=norm, linewidth=1.5)
    lc.set_array(signal[:-1])
    ax5.add_collection(lc)

    ax5.set_xlim(0, time[-1])
    ax5.set_ylim(signal.min() - 2, signal.max() + 2)
    ax5.set_xlabel('Time', color='white')
    ax5.set_ylabel('x(t)', color='white')
    ax5.tick_params(colors='white')
    ax5.spines['bottom'].set_color('white')
    ax5.spines['left'].set_color('white')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.set_title('(e) 1D Observation\n("slow wave" signal)',
                  color='white', fontsize=12, pad=10)

    # === Panel F: Delay-Embedded Reconstruction ===
    ax6 = fig.add_subplot(2, 3, 6, projection='3d', facecolor='black')

    # Delay embedding
    tau = 8
    m = 3
    signal_full = traj[:, 0]
    n_points = len(signal_full) - (m-1) * tau
    embedded = np.zeros((n_points, m))
    for i in range(m):
        embedded[:, i] = signal_full[i*tau : i*tau + n_points]

    # Plot with same color scheme
    points = embedded.reshape(-1, 1, 3)
    colors = plt.cm.plasma(np.linspace(0, 1, len(embedded)-1))
    for i in range(len(embedded)-1):
        ax6.plot3D([embedded[i, 0], embedded[i+1, 0]],
                   [embedded[i, 1], embedded[i+1, 1]],
                   [embedded[i, 2], embedded[i+1, 2]],
                   color=colors[i], linewidth=0.5, alpha=0.8)

    ax6.set_facecolor('black')
    ax6.xaxis.pane.fill = False
    ax6.yaxis.pane.fill = False
    ax6.zaxis.pane.fill = False
    ax6.xaxis.pane.set_edgecolor('gray')
    ax6.yaxis.pane.set_edgecolor('gray')
    ax6.zaxis.pane.set_edgecolor('gray')
    ax6.tick_params(colors='white')
    ax6.set_xlabel('x(t)', color='white')
    ax6.set_ylabel(f'x(t-{tau})', color='white')
    ax6.set_zlabel(f'x(t-{2*tau})', color='white')
    ax6.set_title('(f) Takens Reconstruction\nvia delay embedding',
                  color='white', fontsize=12, pad=10)

    plt.tight_layout()
    plt.savefig('fern_purkinje_comparison.pdf', bbox_inches='tight',
                facecolor='black', dpi=300)
    plt.savefig('fern_purkinje_comparison.png', bbox_inches='tight',
                facecolor='black', dpi=300)
    print("Saved: fern_purkinje_comparison.pdf and .png")


def create_purkinje_detail():
    """Create a detailed Purkinje cell visualization with parallel fibers."""

    fig, ax = plt.subplots(figsize=(12, 14), facecolor='black')
    ax.set_facecolor('black')

    # Generate larger, more detailed arbor
    segments = generate_purkinje_arbor(depth=10, angle_spread=28, length_decay=0.72)
    max_delay = max(s[2] for s in segments)

    # Draw parallel fibers (horizontal, in background)
    n_fibers = 40
    for i in range(n_fibers):
        y_pos = 5 + i * 1.3
        length = 60
        delay_color = DELAY_CMAP(i / n_fibers)
        ax.plot([-30, 30], [y_pos, y_pos], color=delay_color,
                linewidth=0.5, alpha=0.3, zorder=1)

    # Draw dendritic arbor
    for (x1, y1), (x2, y2), delay, depth in segments:
        color = DELAY_CMAP(delay / max_delay)
        linewidth = 0.2 + depth * 0.25
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth,
                alpha=0.9, zorder=2)

    # Draw synaptic contacts where fibers cross dendrites
    for (x1, y1), (x2, y2), delay, depth in segments:
        if depth > 2:  # Only on finer branches
            for i in range(n_fibers):
                fiber_y = 5 + i * 1.3
                if min(y1, y2) < fiber_y < max(y1, y2):
                    # Interpolate x position
                    if y2 != y1:
                        t = (fiber_y - y1) / (y2 - y1)
                        x_cross = x1 + t * (x2 - x1)
                        if -25 < x_cross < 25:
                            ax.plot(x_cross, fiber_y, 'o',
                                   color=DELAY_CMAP(i/n_fibers),
                                   markersize=1.5, alpha=0.7, zorder=3)

    # Draw soma
    soma = Circle((0, -3), 4, color=PURKINJE_PURPLE, alpha=0.9, zorder=4)
    ax.add_patch(soma)
    ax.text(0, -3, 'Purkinje\nSoma', ha='center', va='center',
            color='white', fontsize=10, fontweight='bold')

    # Draw axon
    ax.plot([0, 0], [-7, -20], color=PURKINJE_PURPLE, linewidth=3, zorder=4)
    ax.annotate('', xy=(0, -25), xytext=(0, -20),
                arrowprops=dict(arrowstyle='->', color=PURKINJE_PURPLE, lw=2))
    ax.text(0, -28, 'Output to\ndeep nuclei', ha='center', va='top',
            color='white', fontsize=10)

    # Add annotations
    ax.text(-28, 35, 'Parallel fibers\n(delay gradient)',
            ha='left', va='center', color='white', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='black',
                     edgecolor='gray', alpha=0.8))

    ax.annotate('', xy=(-25, 25), xytext=(-25, 45),
                arrowprops=dict(arrowstyle='<->', color='white', lw=1.5))
    ax.text(-27, 35, 'τ', ha='right', va='center', color='white', fontsize=14)

    # Title
    ax.text(0, 62, 'Purkinje Cell as Delay-Coordinate Sampler',
            ha='center', va='bottom', color='white', fontsize=14, fontweight='bold')
    ax.text(0, 58, 'Each parallel fiber arrives with different delay\n'
            '→ dendritic arbor samples across delay-coordinate space',
            ha='center', va='top', color='lightgray', fontsize=11)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=DELAY_CMAP, norm=plt.Normalize(0, 1))
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02,
                       label='Relative delay')
    cbar.set_label('Conduction delay', color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    ax.set_xlim(-35, 35)
    ax.set_ylim(-32, 65)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('purkinje_detail.pdf', bbox_inches='tight',
                facecolor='black', dpi=300)
    plt.savefig('purkinje_detail.png', bbox_inches='tight',
                facecolor='black', dpi=300)
    print("Saved: purkinje_detail.pdf and .png")


def create_fern_detail():
    """Create a beautiful fern with annotations about spatial sampling."""

    fig, ax = plt.subplots(figsize=(10, 14), facecolor='black')
    ax.set_facecolor('black')

    # Generate high-resolution fern
    fern = barnsley_fern(150000)

    # Create gradient based on position
    # Color by both height and horizontal spread
    colors = np.zeros((len(fern), 4))
    height_norm = fern[:, 1] / 10
    spread_norm = (np.abs(fern[:, 0]) + 0.5) / 3

    # Green gradient with golden accents at tips
    for i, (x, y) in enumerate(fern):
        h = np.clip(y / 10, 0, 1)
        s = np.clip(np.abs(x) / 3, 0, 1)
        # Base green
        r = 0.1 + 0.2 * h
        g = 0.4 + 0.5 * h
        b = 0.1 + 0.1 * h
        # Add golden tint at edges
        if s > 0.3:
            r += 0.3 * s
            g += 0.2 * s
        colors[i] = [np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1), 0.7]

    ax.scatter(fern[:, 0], fern[:, 1], c=colors, s=0.2)

    # Add light rays to show spatial sampling
    for angle in np.linspace(-30, 30, 7):
        x_start = -4
        y_start = 12
        x_end = x_start + 8 * np.sin(np.radians(angle + 90))
        y_end = y_start - 8 * np.cos(np.radians(angle + 90))
        ax.plot([x_start, x_end], [y_start, y_end],
                color='yellow', alpha=0.15, linewidth=20)
        ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                   arrowprops=dict(arrowstyle='->', color='yellow',
                                  lw=1, alpha=0.5))

    # Annotations
    ax.text(0, 11.5, 'Fractal Geometry Optimizes\nSpatial Sampling',
            ha='center', va='bottom', color='white', fontsize=14, fontweight='bold')

    ax.text(3, 7, 'Each frond samples\ndifferent light angles',
            ha='left', va='center', color='lightgreen', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='black',
                     edgecolor='green', alpha=0.8))

    ax.text(-3, 3, 'Self-similar branching\nmaximizes coverage',
            ha='right', va='center', color='lightgreen', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='black',
                     edgecolor='green', alpha=0.8))

    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.5, 12)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('fern_detail.pdf', bbox_inches='tight',
                facecolor='black', dpi=300)
    plt.savefig('fern_detail.png', bbox_inches='tight',
                facecolor='black', dpi=300)
    print("Saved: fern_detail.pdf and .png")


def create_convergent_evolution_summary():
    """Side-by-side comparison showing the geometric principle."""

    fig = plt.figure(figsize=(16, 8), facecolor='black')

    # Left: Fern
    ax1 = fig.add_subplot(1, 3, 1, facecolor='black')
    fern = barnsley_fern(80000)
    colors = plt.cm.YlGn(np.clip(fern[:, 1] / 10, 0, 1))
    ax1.scatter(fern[:, 0], fern[:, 1], c=colors, s=0.15, alpha=0.8)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(0, 10.5)
    ax1.axis('off')
    ax1.set_title('FERN\n\nSamples: Light angles\nDomain: 3D space\nAge: ~360 Mya',
                  color='lightgreen', fontsize=12, pad=10)

    # Center: Principle
    ax2 = fig.add_subplot(1, 3, 2, facecolor='black')
    ax2.axis('off')

    # Draw the key insight
    ax2.text(0.5, 0.85, 'LOW-D IN HIGH-D',
             ha='center', va='center', transform=ax2.transAxes,
             color='white', fontsize=16, fontweight='bold')

    ax2.text(0.5, 0.7, 'Same Geometry\nDifferent Mechanisms',
             ha='center', va='center', transform=ax2.transAxes,
             color='gold', fontsize=14)

    # Draw connecting arrows
    ax2.annotate('', xy=(0.2, 0.5), xytext=(0.5, 0.6),
                arrowprops=dict(arrowstyle='->', color='white', lw=2),
                transform=ax2.transAxes)
    ax2.annotate('', xy=(0.8, 0.5), xytext=(0.5, 0.6),
                arrowprops=dict(arrowstyle='->', color='white', lw=2),
                transform=ax2.transAxes)

    # The principle
    box_text = """
    THE GEOMETRIC PRINCIPLE

    When you need to sample a
    high-dimensional input space:

    → Grow a fractal tree

    • Maximizes coverage
    • Self-similar at all scales
    • Metabolically efficient
    """
    ax2.text(0.5, 0.25, box_text, ha='center', va='center',
             transform=ax2.transAxes, color='white', fontsize=11,
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#1a1a2e',
                      edgecolor='gold', linewidth=2))

    # Right: Purkinje
    ax3 = fig.add_subplot(1, 3, 3, facecolor='black')
    segments = generate_purkinje_arbor(depth=9, angle_spread=30)
    max_delay = max(s[2] for s in segments)

    for (x1, y1), (x2, y2), delay, depth in segments:
        color = DELAY_CMAP(delay / max_delay)
        linewidth = 0.2 + depth * 0.25
        ax3.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=0.9)

    soma = Circle((0, -2), 3, color=PURKINJE_PURPLE, alpha=0.9)
    ax3.add_patch(soma)

    ax3.set_xlim(-25, 25)
    ax3.set_ylim(-8, 55)
    ax3.axis('off')
    ax3.set_title('PURKINJE CELL\n\nSamples: Time delays\nDomain: State space\nAge: ~500 Mya',
                  color=PURKINJE_LIGHT, fontsize=12, pad=10)

    plt.tight_layout()
    plt.savefig('convergent_evolution.pdf', bbox_inches='tight',
                facecolor='black', dpi=300)
    plt.savefig('convergent_evolution.png', bbox_inches='tight',
                facecolor='black', dpi=300)
    print("Saved: convergent_evolution.pdf and .png")


if __name__ == "__main__":
    print("Generating fern/Purkinje visualizations...\n")

    create_main_figure()
    create_purkinje_detail()
    create_fern_detail()
    create_convergent_evolution_summary()

    print("\nAll figures generated!")
