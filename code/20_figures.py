"""Figures 1-4 for manuscript v4. Output: SVG with live text (editable) + PNG fallback."""
import json, sys
import numpy as np
import matplotlib, logging; logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "outputs" / "sim_numbers.json")
OUT = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "outputs" / "figures") + "/"
SPLIT = json.load(open(ROOT / "outputs" / "out_08_split_distributions.json"))
D = json.load(open(SRC))
BLUE, ORANGE, GRAY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#8a8985", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]          # one hue, light -> dark
plt.rcParams.update({
    "svg.fonttype": "none", "font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.6,
    "axes.titlesize": 9, "axes.titleweight": "bold", "legend.frameon": False, "legend.fontsize": 7.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "savefig.dpi": 300,
})
DOMS = [("Communication", "Communication"), ("Energy", "Energy"), ("Healthcare", "Healthcare")]

# Figure 1a uses the output of the original analysis script (02_resampling.py) so that figure and text share one null definition
import csv, os
CURVE = os.environ.get("ALPHA_CURVE", str(ROOT / "outputs" / "out_02_resampling_curve.csv"))
if os.path.exists(CURVE):
    for row in csv.DictReader(open(CURVE)):
        for x in D["A_subsets"][row["Domain"]]:
            if x["n"] == int(row["n"]):
                assert abs(x["a_mean"] - float(row["obs_mean"])) < 1e-6
                x.update(a_lo=float(row["obs_lo"]), a_hi=float(row["obs_hi"]), null_mean=float(row["null_mean"]), null_lo=float(row["null_lo"]), null_hi=float(row["null_hi"]))


def save(fig, name):
    fig.savefig(OUT + name + ".svg", bbox_inches="tight")
    fig.savefig(OUT + name + ".png", bbox_inches="tight")
    plt.close(fig)


# ---------- Figure 1a: alpha vs panel size, with permutation null
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
for ax, (dom, lab) in zip(axs, DOMS):
    r = D["A_subsets"][dom]; n = [x["n"] for x in r]
    ax.plot(n, [x["null_lo"] for x in r], ":", color=GRAY, lw=1.0)
    ax.plot(n, [x["null_hi"] for x in r], ":", color=GRAY, lw=1.0)
    ax.plot(n, [x["null_mean"] for x in r], "--", color=GRAY, lw=1.4, label="Permutation null: mean (dashed), 95% interval (dotted)")
    ax.plot(n, [x["a_lo"] for x in r], "-", color=BLUE, lw=0.7)
    ax.plot(n, [x["a_hi"] for x in r], "-", color=BLUE, lw=0.7)
    ax.plot(n, [x["a_mean"] for x in r], "-o", color=BLUE, ms=3.5, label="Observed subsets: mean (bold), 95% interval (thin)")
    ax.axhline(0.667, color=INK, ls=":", lw=0.9)
    ax.set_title(lab); ax.set_xlabel("Raters in subset (k)"); ax.set_xticks(n); ax.set_ylim(-1.0, 0.8)
axs[0].set_ylabel("Krippendorff's α")
axs[0].text(2.1, 0.69, "α = .667", fontsize=7.5, color=INK, va="bottom")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.09))
fig.tight_layout(); fig.text(0.005, 0.99, "(a)", fontsize=10, fontweight="bold", color=INK, va="top"); save(fig, "fig1a_alpha_by_panel_size")

# ---------- Figure 1b: ICC(C,1), ICC(C,k), Spearman-Brown
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
for ax, (dom, lab) in zip(axs, DOMS):
    r = D["A_subsets"][dom]; n = [x["n"] for x in r]
    ax.plot(n, [x["sb"] for x in r], "--", color=ORANGE, lw=1.4, label="Spearman–Brown prediction from full-panel ICC(C,1)")
    ax.plot(n, [x["c1"] for x in r], "-s", color=GRAY, ms=3, label="ICC(C,1), subset mean")
    ax.plot(n, [x["ck"] for x in r], "-o", color=BLUE, ms=3.5, label="ICC(C,k), subset mean")
    ax.axhline(0.70, color=INK, ls=":", lw=0.9)
    ax.set_title(lab); ax.set_xlabel("Raters in subset (k)"); ax.set_xticks(n); ax.set_ylim(-0.1, 0.8)
axs[0].set_ylabel("Intraclass correlation (consistency)")
axs[0].text(2.1, 0.715, "target = .70", fontsize=7.5, color=INK, va="bottom")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
fig.tight_layout(); fig.text(0.005, 0.99, "(b)", fontsize=10, fontweight="bold", color=INK, va="top"); save(fig, "fig1b_icc_by_panel_size")

# ---------- Figure 2: split-half distribution of cross-panel rho
bins = np.array(SPLIT["bins"])
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
for ax, (dom, lab) in zip(axs, DOMS):
    b = SPLIT["dom"][dom]
    hgt = np.array(b["hist"], float); hgt = hgt / hgt.sum()
    ax.stairs(hgt, bins, color=GRAY, lw=1.2, fill=True, facecolor="#ecebe7", label="All splits at the actual panel sizes")
    ax.axvline(b["mean"], color=GRAY, ls="--", lw=1.4, label="Mean of splits")
    ax.axvline(b["pred"], color=ORANGE, ls=":", lw=1.6, label="Attenuation prediction")
    ax.axvline(b["obs"], color=BLUE, lw=2.0, label="Observed (first vs. repeat panel)")
    ax.set_title(f"{lab} ({b['kA']}/{b['kB']}, {b['n_splits']} splits)", fontsize=8.5)
    ax.set_xlabel("Cross-panel Spearman ρ"); ax.set_xlim(bins[0], bins[-1]); ax.grid(axis="x", visible=False)
axs[0].set_ylabel("Share of splits")
ymax = max(a.get_ylim()[1] for a in axs) * 1.22
for a in axs: a.set_ylim(0, ymax)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07), columnspacing=1.2)
fig.tight_layout(); save(fig, "fig2_split_half_distribution")

# ---------- Figure 3: what moves alpha vs. what moves rho
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax = axs[0]
S = D["D_sim2a_crn"]; rows = S["rows"]; x = np.arange(len(rows))
ax.axvspan(5.9, 7.1, color="#f1f0ec", zorder=0)
ax.text(6.5, 0.345, "this study\n(69–99)", ha="center", va="top", fontsize=7.5, color=MUTED)
ax.plot(x, [r["rho_mean"] for r in rows], "-", color=ORANGE, label="Cross-panel ρ, 5 vs. 5 raters (simulated)")
ax.plot(x, [r["alpha_cf"] for r in rows], "-", color=BLUE, label="α, 10 raters (closed form)")
ax.plot(x, [r["alpha_sim"] for r in rows], "o", color=BLUE, ms=4, mec="white", mew=0.6, label="α, 10 raters (simulated)")
ax.axhline(S["alpha_floor"], color=BLUE, ls=":", lw=1.0)
ax.text(0.0, S["alpha_floor"] - 0.012, "lower limit of α = −.104", fontsize=7.5, color=INK, va="top")
ax.axhline(0, color=MUTED, lw=0.5)
ax.set_xticks(x); ax.set_xticklabels([str(r["R"]) for r in rows]); ax.set_ylim(-0.16, 0.36)
ax.set_xlabel("Severity variance / target variance"); ax.set_ylabel("Expected value"); ax.set_title("(a) Severity lowers α, not ρ")
ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.61))
ax = axs[1]
for c, col in zip(D["E_curves"], SEQ[::-1]):
    ks = [p["k"] for p in c["curve"]]
    ax.plot(ks, [p["spearman"] for p in c["curve"]], "-", color=col)
    sim = c["sim"]; skey = [k for k in sim[0] if k != "k"][0]
    ax.plot([p["k"] for p in sim], [p[skey] for p in sim], "o", color=col, ms=4, mec="white", mew=0.6)
    ax.text(ks[-1] + 0.6, c["curve"][-1]["spearman"], f"{c['ve_over_vt']:.0f}", fontsize=7.5, color=INK, va="center")
ax.text(1.02, 1.0, "σe²/σt²", transform=ax.transAxes, fontsize=7.5, color=INK, va="bottom", ha="left")
ax.plot([], [], "-", color=SEQ[2], label="Attenuation prediction (rank-corrected)"); ax.plot([], [], "o", color=SEQ[2], ms=4, label="Simulated mean")
ax.axhline(0.5, color=INK, ls=":", lw=0.9)
ax.set_xlabel("Raters per panel (k)"); ax.set_ylabel("Expected cross-panel Spearman ρ"); ax.set_ylim(0, 0.9); ax.set_xlim(1, 31)
ax.set_title("(b) Residual ratio and panel size set ρ"); ax.legend(loc="upper left")
fig.tight_layout(); save(fig, "fig3_alpha_vs_rho_conditions")

# ---------- Figure 4: decision accuracy by panel size
C = D["F_decisions"]["curves"]; ks = [c["k"] for c in C]
fig, ax = plt.subplots(figsize=(4.6, 3.1))
ax.plot(ks, [c["cmp_50"] for c in C], "-", color=BLUE, label="Comparative, Δ = 0.50")
ax.plot(ks, [c["thr_50"] for c in C], "--", color=BLUE, label="Threshold, Δ = 0.50")
ax.plot(ks, [c["cmp_25"] for c in C], "-", color=ORANGE, label="Comparative, Δ = 0.25")
ax.plot(ks, [c["thr_25"] for c in C], "--", color=ORANGE, label="Threshold, Δ = 0.25")
ax.axhline(0.90, color=INK, ls=":", lw=0.9); ax.text(39.5, 0.885, ".90", fontsize=7.5, color=INK, va="top", ha="right")
ax.set_xlabel("Raters in panel (k)"); ax.set_ylabel("Probability of a correct decision"); ax.set_ylim(0.5, 1.005); ax.set_xlim(1, 40)
ax.legend(loc="lower right")
fig.tight_layout(); save(fig, "fig4_decision_accuracy")
print("ok")
