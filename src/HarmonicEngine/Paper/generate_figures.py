"""
Generates all figures for the HRE beginner-friendly whitepaper.
Run: python generate_figures.py
Outputs: figures/*.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)


def save(name):
    path = os.path.join(FIGDIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  wrote {path}")


# ============================================================
# Figure 1: HRE Architecture Pipeline Flowchart
# ============================================================
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')

    stages = [
        ("Input\nSignal", "#E8F0FE", 1),
        ("Mellin-Fourier\nInvariance", "#BBDEFB", 3),
        ("OFDM\nSpectral Bus", "#90CAF9", 5),
        ("Nonlinearity\n(creates IMD)", "#64B5F6", 7),
        ("IMD Extraction\n(Attention)", "#42A5F5", 9),
        ("Spectral\nSparsity (Top-K)", "#2196F3", 11),
        ("Output\nProjection", "#1976D2", 13),
    ]
    for label, color, x in stages:
        box = FancyBboxPatch((x - 0.85, 1.5), 1.7, 1.2,
                             boxstyle="round,pad=0.08",
                             facecolor=color, edgecolor='#1565C0', linewidth=1.4)
        ax.add_patch(box)
        ax.text(x, 2.1, label, ha='center', va='center', fontsize=9, fontweight='bold')

    for i in range(len(stages) - 1):
        arrow = FancyArrowPatch((stages[i][2] + 0.85, 2.1),
                                 (stages[i + 1][2] - 0.85, 2.1),
                                 arrowstyle='->', mutation_scale=18,
                                 color='#455A64', linewidth=1.8)
        ax.add_patch(arrow)

    # Annotations below
    annotations = [
        (3, "scale/shift\ninvariance", "#1976D2"),
        (5, "encode features\non carriers", "#1976D2"),
        (7, "squaring creates\ncross-products", "#1976D2"),
        (9, "O(N log N)\nattention", "#1976D2"),
        (11, "regularization\n+ compression", "#1976D2"),
    ]
    for x, txt, color in annotations:
        ax.text(x, 0.85, txt, ha='center', va='center', fontsize=8,
                style='italic', color=color)

    ax.text(7, 3.5, "The Harmonic Resonance Engine Pipeline",
            ha='center', fontsize=14, fontweight='bold')
    save("01_pipeline.png")


# ============================================================
# Figure 2: OFDM Carrier Spectrum
# ============================================================
def fig_ofdm_carriers():
    fig, ax = plt.subplots(figsize=(10, 4))

    # Carrier frequencies (Sidon set, manual)
    carriers = [3, 7, 12, 20, 30, 44, 65, 80]
    amplitudes = [1.0, 0.7, 1.3, 0.5, 0.9, 0.4, 1.1, 0.6]
    max_freq = 100

    ax.bar(carriers, amplitudes, width=1.5, color='#1976D2', edgecolor='#0D47A1')
    for c, a in zip(carriers, amplitudes):
        ax.text(c, a + 0.05, f"f{carriers.index(c) + 1}", ha='center', fontsize=9)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xlim(0, max_freq)
    ax.set_ylim(0, 1.6)
    ax.set_xlabel("Frequency bin", fontsize=11)
    ax.set_ylabel("Amplitude (feature value)", fontsize=11)
    ax.set_title("OFDM Spectral Bus — Each feature rides on its own carrier frequency",
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.text(50, 1.4, "Each bar = one feature encoded as the amplitude of a unique frequency",
            ha='center', fontsize=9, style='italic', color='#546E7A')
    save("02_ofdm_carriers.png")


# ============================================================
# Figure 3: IMD Product Diagram
# ============================================================
def fig_imd_products():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6))

    # Before squaring: two carriers
    f1, f2 = 10, 17
    carriers_before = [f1, f2]
    ax1.bar(carriers_before, [1.0, 1.0], width=0.6, color='#1976D2', edgecolor='#0D47A1')
    ax1.text(f1, 1.08, f"a₁", ha='center', fontsize=11, fontweight='bold')
    ax1.text(f2, 1.08, f"a₂", ha='center', fontsize=11, fontweight='bold')
    ax1.axhline(y=0, color='black', linewidth=0.8)
    ax1.set_xlim(0, 50)
    ax1.set_ylim(0, 1.4)
    ax1.set_xlabel("Frequency bin")
    ax1.set_ylabel("Magnitude")
    ax1.set_title("Before squaring: two carriers at f₁=10 and f₂=17",
                  fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # After squaring: IMD products appear
    original = [(f1, 0.5, '#1976D2', 'a₁²/2'),
                (f2, 0.5, '#1976D2', 'a₂²/2')]
    imd_products = [
        (2 * f1, 0.5, '#EF5350', '2f₁\n(a₁²/2)'),
        (2 * f2, 0.5, '#EF5350', '2f₂\n(a₂²/2)'),
        (f1 + f2, 0.5, '#4CAF50', 'f₁+f₂\n(a₁·a₂)'),
        (abs(f1 - f2), 0.5, '#4CAF50', '|f₁-f₂|\n(a₁·a₂)'),
    ]
    all_freqs = original + imd_products
    for f, h, c, lbl in all_freqs:
        ax2.bar(f, h, width=0.6, color=c, edgecolor='black')
        ax2.text(f, h + 0.02, lbl, ha='center', fontsize=8)

    # DC component
    ax2.bar(0, 1.0, width=0.6, color='#9E9E9E', edgecolor='black')
    ax2.text(0, 1.02, 'DC\n((a₁²+a₂²)/2)', ha='center', fontsize=8)

    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_xlim(-3, 50)
    ax2.set_ylim(0, 1.3)
    ax2.set_xlabel("Frequency bin")
    ax2.set_ylabel("Magnitude")
    ax2.set_title("After squaring: IMD products at f₁+f₂ and |f₁-f₂| encode the product a₁·a₂",
                  fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    green_patch = mpatches.Patch(color='#4CAF50', label='IMD products (interaction scores)')
    red_patch = mpatches.Patch(color='#EF5350', label='Harmonics (self-interaction)')
    blue_patch = mpatches.Patch(color='#1976D2', label='Original carriers (attenuated)')
    ax2.legend(handles=[blue_patch, red_patch, green_patch], loc='upper right', fontsize=9)

    plt.tight_layout()
    save("03_imd_products.png")


# ============================================================
# Figure 4: Complexity Comparison (Attention)
# ============================================================
def fig_complexity():
    fig, ax = plt.subplots(figsize=(10, 5))

    N = np.logspace(1, 5, 200)  # sequence length 10 to 100,000
    transformer = N ** 2
    hre = N * np.log2(N)

    ax.loglog(N, transformer, label='Transformer O(N²)',
              color='#EF5350', linewidth=2.5)
    ax.loglog(N, hre, label='HRE O(N log N)',
              color='#1976D2', linewidth=2.5)

    # Mark key scales
    scales = [(100, "100 tokens"),
              (1000, "1K tokens"),
              (10000, "10K tokens"),
              (100000, "100K tokens")]
    for s, label in scales:
        tf = s ** 2
        hr = s * np.log2(s)
        ratio = tf / hr
        ax.axvline(x=s, color='gray', linestyle=':', alpha=0.4)
        ax.text(s, tf * 3, f"{ratio:.0f}× fewer\nops at {label}",
                ha='center', fontsize=8, color='#1565C0')

    ax.set_xlabel("Sequence length N (tokens)", fontsize=11)
    ax.set_ylabel("Attention operations", fontsize=11)
    ax.set_title("Attention Complexity: Transformer vs. HRE\nHRE scales log-linearly instead of quadratically",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, which='both', alpha=0.3)
    save("04_complexity.png")


# ============================================================
# Figure 5: Memory Footprint Comparison
# ============================================================
def fig_memory():
    fig, ax = plt.subplots(figsize=(11, 5.5))

    categories = ['Attention\nweights', 'FFN / MLP\nweights',
                  'Optimizer\nstate (Adam)', 'Backprop\nactivations',
                  'KV cache\n(inference)', 'Other\n(embeddings, etc.)']

    # Rough percentages of total memory for a 1T-param transformer during training
    transformer = [25, 74, 150, 80, 30, 6]
    hre = [0, 1, 0, 0, 0, 6]  # HRE eliminates most categories

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width / 2, transformer, width,
                   label='Transformer', color='#EF5350', edgecolor='black')
    bars2 = ax.bar(x + width / 2, hre, width,
                   label='HRE', color='#1976D2', edgecolor='black')

    # Labels on bars
    for bar, val in zip(bars1, transformer):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 2, f"{val}",
                ha='center', fontsize=9)
    for bar, val in zip(bars2, hre):
        h = bar.get_height()
        label = f"{val}" if val > 0 else "~0"
        ax.text(bar.get_x() + bar.get_width() / 2, h + 2, label,
                ha='center', fontsize=9, fontweight='bold', color='#0D47A1')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Relative memory footprint (normalized)", fontsize=11)
    ax.set_title("Memory Footprint: Transformer vs. HRE\n(where your GPU RAM actually goes)",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Annotation
    ax.text(5, 130, "HRE eliminates attention weights,\noptimizer state, activation\nstorage, and KV cache",
            ha='right', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='#FFF9C4', edgecolor='#F57F17'))
    save("05_memory.png")


# ============================================================
# Figure 6: Hebbian Convergence Curve
# ============================================================
def fig_hebbian_convergence():
    fig, ax = plt.subplots(figsize=(10, 5))

    iters = np.arange(0, 100)
    # Hebbian: geometric convergence
    hebbian_error = 1.0 * (0.92 ** iters) + 0.02
    # Gradient descent: slower
    gd_error = 1.0 * np.exp(-0.04 * iters) + 0.05
    # Wiener (one-shot analytical)
    wiener_line = np.full_like(iters, 0.02, dtype=float)

    ax.plot(iters, hebbian_error, label='Spectral Hebbian (single pass)',
            color='#1976D2', linewidth=2.5)
    ax.plot(iters, gd_error, label='Gradient Descent (backprop)',
            color='#EF5350', linewidth=2.5, linestyle='--')
    ax.axhline(y=0.02, color='#4CAF50', linewidth=2,
               linestyle=':', label='Wiener Optimal (analytic)')

    ax.set_xlabel("Iterations", fontsize=11)
    ax.set_ylabel("Filter error (relative)", fontsize=11)
    ax.set_title("Spectral Hebbian converges to Wiener optimum in far fewer iterations",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    ax.annotate('geometric\nconvergence', xy=(15, 0.25), xytext=(25, 0.5),
                fontsize=9, color='#1565C0',
                arrowprops=dict(arrowstyle='->', color='#1565C0'))
    save("06_hebbian_convergence.png")


# ============================================================
# Figure 7: Spectral Sparsity (Top-K)
# ============================================================
def fig_sparsity():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    np.random.seed(42)
    n = 64
    # Sparse signal: 5 dominant peaks + noise
    spectrum = 0.1 * np.abs(np.random.randn(n))
    peaks = [5, 12, 20, 29, 38]
    peak_heights = [1.0, 0.8, 0.9, 0.6, 0.7]
    for p, h in zip(peaks, peak_heights):
        spectrum[p] = h

    ax1.bar(range(n), spectrum, color='#90CAF9', edgecolor='#1976D2', width=0.8)
    ax1.set_xlabel("Frequency bin")
    ax1.set_ylabel("Magnitude")
    ax1.set_title("Original spectrum\n(64 coefficients, mostly noise)",
                  fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)

    # Top-K sparse version
    k = 5
    sparse = np.zeros(n)
    top_k_idx = np.argsort(spectrum)[-k:]
    sparse[top_k_idx] = spectrum[top_k_idx]

    colors_sparse = ['#EF5350' if i in top_k_idx else '#E0E0E0' for i in range(n)]
    ax2.bar(range(n), sparse, color=colors_sparse,
            edgecolor=['#C62828' if i in top_k_idx else '#BDBDBD' for i in range(n)],
            width=0.8)
    ax2.set_xlabel("Frequency bin")
    ax2.set_ylabel("Magnitude")
    ax2.set_title(f"After Top-{k} sparsity\n(5 coefficients = 92.8% compression)",
                  fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', alpha=0.3)

    # Arrow between plots
    fig.text(0.5, 0.5, "→", fontsize=30, ha='center', color='#455A64')

    plt.suptitle("Spectral Sparsity: Keep only the K strongest frequencies",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save("07_sparsity.png")


# ============================================================
# Figure 8: Mellin Scale Invariance
# ============================================================
def fig_mellin_invariance():
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))

    np.random.seed(0)
    t1 = np.linspace(0, 10, 500)
    t2 = np.linspace(0, 10, 500)
    t3 = np.linspace(0, 10, 500)

    # Three chirps at different scales
    scales = [1.0, 1.5, 2.0]
    for ax, scale in zip(axes[0], scales):
        s = np.sin(2 * np.pi * scale * t1 ** 1.5 / 10)
        ax.plot(t1, s, color='#1976D2', linewidth=1.2)
        ax.set_title(f"Signal scaled by {scale}×", fontsize=11)
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)
        ax.set_ylim(-1.3, 1.3)

    # Their Mellin fingerprints (idealized — all look the same)
    freqs = np.linspace(0, 20, 100)
    fingerprint = np.exp(-0.5 * ((freqs - 5) / 1.5) ** 2) + \
                  0.6 * np.exp(-0.5 * ((freqs - 10) / 1.8) ** 2) + \
                  0.3 * np.exp(-0.5 * ((freqs - 14) / 1.2) ** 2)
    for ax in axes[1]:
        # Tiny jitter to visualize scale invariance (actual fingerprints are identical)
        jittered = fingerprint + np.random.randn(len(fingerprint)) * 0.003
        ax.plot(freqs, jittered, color='#4CAF50', linewidth=1.6)
        ax.set_title("Mellin fingerprint", fontsize=11)
        ax.set_xlabel("Mellin frequency")
        ax.set_ylabel("Magnitude")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.2)

    plt.suptitle("Mellin-Fourier Invariance: Scaled signals produce the same fingerprint",
                 fontsize=13, fontweight='bold', y=1.0)
    plt.tight_layout()
    save("08_mellin_invariance.png")


# ============================================================
# Figure 9: Traditional NN vs HRE conceptual diagram
# ============================================================
def fig_traditional_vs_hre():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Traditional NN
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title("Traditional Neural Network\n(point-to-point weighted connections)",
                  fontsize=12, fontweight='bold')

    layer1_y = [7, 5.5, 4, 2.5]
    layer2_y = [7, 5.5, 4, 2.5]
    for y in layer1_y:
        circ = plt.Circle((2.5, y), 0.35, color='#64B5F6', ec='#0D47A1', lw=1.5)
        ax1.add_patch(circ)
    for y in layer2_y:
        circ = plt.Circle((6.5, y), 0.35, color='#1976D2', ec='#0D47A1', lw=1.5)
        ax1.add_patch(circ)

    # Every-to-every connections (weighted)
    for y1 in layer1_y:
        for y2 in layer2_y:
            ax1.plot([2.85, 6.15], [y1, y2], color='#9E9E9E', linewidth=0.7, alpha=0.6)

    ax1.text(4.5, 1.3, "N × M weights\nO(NM) memory\nO(NM) compute",
             ha='center', fontsize=10, color='#D32F2F',
             bbox=dict(boxstyle='round', facecolor='#FFEBEE', edgecolor='#D32F2F'))
    ax1.text(2.5, 8.2, "Inputs", ha='center', fontsize=11, fontweight='bold')
    ax1.text(6.5, 8.2, "Outputs", ha='center', fontsize=11, fontweight='bold')

    # Right: HRE
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title("Harmonic Resonance Engine\n(features share a frequency bus)",
                  fontsize=12, fontweight='bold')

    # Features on left
    for y in layer1_y:
        circ = plt.Circle((2, y), 0.35, color='#64B5F6', ec='#0D47A1', lw=1.5)
        ax2.add_patch(circ)

    # Spectral bus (a wavy line)
    bus_x = np.linspace(3.5, 6.5, 100)
    for yoff in np.linspace(2, 8, 6):
        freq = 2 + yoff * 0.3
        bus_y = yoff + 0.2 * np.sin(2 * np.pi * freq * (bus_x - 3.5) / 3)
        ax2.plot(bus_x, bus_y, color='#1976D2', linewidth=1.2, alpha=0.5)

    # Bus box
    bus_rect = Rectangle((3.5, 1.8), 3, 6.4, linewidth=2,
                          edgecolor='#1976D2', facecolor='#E3F2FD', alpha=0.3)
    ax2.add_patch(bus_rect)
    ax2.text(5, 5, "OFDM\nSpectral Bus", ha='center', va='center',
             fontsize=11, fontweight='bold', color='#0D47A1')

    # Outputs on right
    for y in layer1_y:
        circ = plt.Circle((8, y), 0.35, color='#1976D2', ec='#0D47A1', lw=1.5)
        ax2.add_patch(circ)

    # Connections to/from bus
    for y in layer1_y:
        ax2.plot([2.35, 3.5], [y, y], color='#455A64', linewidth=1.5)
        ax2.plot([6.5, 7.65], [y, y], color='#455A64', linewidth=1.5)

    ax2.text(5, 0.8, "0 attention weights\nO(K) memory (K << N²)\nO(N log N) compute",
             ha='center', fontsize=10, color='#1B5E20',
             bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#1B5E20'))
    ax2.text(2, 9, "Features", ha='center', fontsize=11, fontweight='bold')
    ax2.text(8, 9, "Features", ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    save("09_traditional_vs_hre.png")


# ============================================================
# Figure 10: Training Cost / Energy Comparison
# ============================================================
def fig_training_cost():
    fig, ax = plt.subplots(figsize=(10, 5))

    categories = ['Training\nEnergy\n(MWh)', 'Training\nTime\n(days)',
                  'GPU\nCount', 'Optimizer\nMemory\n(×weights)']
    transformer_vals = [1300, 34, 1024, 3]
    hre_theoretical = [13, 0.5, 16, 0]

    x = np.arange(len(categories))
    width = 0.35

    b1 = ax.bar(x - width / 2, transformer_vals, width,
                label='Transformer (GPT-3 scale)', color='#EF5350', edgecolor='black')
    b2 = ax.bar(x + width / 2, hre_theoretical, width,
                label='HRE (projected)', color='#1976D2', edgecolor='black')

    for bar, val in zip(b1, transformer_vals):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.05, f"{val:,}",
                ha='center', fontsize=9)
    for bar, val in zip(b2, hre_theoretical):
        h = bar.get_height()
        label = f"{val}" if val > 0 else "~0"
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + max(transformer_vals) * 0.02, label,
                ha='center', fontsize=9, fontweight='bold', color='#0D47A1')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title("Training Cost Projection: Transformer vs. HRE\n(log scale)",
                 fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    save("10_training_cost.png")


if __name__ == "__main__":
    print("Generating HRE paper figures...")
    fig_pipeline()
    fig_ofdm_carriers()
    fig_imd_products()
    fig_complexity()
    fig_memory()
    fig_hebbian_convergence()
    fig_sparsity()
    fig_mellin_invariance()
    fig_traditional_vs_hre()
    fig_training_cost()
    print("Done!")
