"""Figures and the numbers quoted in the paper, computed from the records in results/eval.

    uv run python paper/figs.py

Writes paper/figures/*.pdf (and tradeoff.png for the README) and prints every pooled number the text uses.
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from learned_selection.curves import bootstrap_ci, load
from learned_selection.instances import FRESH_SEEDS, HELD_OUT_SETTINGS, TRAIN_SETTINGS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"
OUT.mkdir(exist_ok=True)
EVAL = ROOT / "results" / "eval"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, PALE = "#1b1b1a", "#52514e", "#8d8c88", "#e4e3df", "#c9c8c3"
COL_W, PAGE_W = 3.375, 7.0   # REVTeX column and page widths, inches

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "xtick.minor.width": 0.4,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

S8 = "bld-h3+sage-h4+dsage-h5+mcts-c0.5"
B8 = "bld-h3+dgin-h4+dgin-h5"
L = "lin-h2h4+sage-h5"
D = "dgin-h2h4h5"
M = "dgin-h2+sage-h4+dsage-h5+mcts-c0.5"
TRAIN = {(f, json.dumps(p, sort_keys=True)) for f, p in TRAIN_SETTINGS}
HELD = {(f, json.dumps(p, sort_keys=True)) for f, p in HELD_OUT_SETTINGS}


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"wrote {OUT / name}")


def files(*patterns):
    return sorted(f for p in patterns for f in glob.glob(str(EVAL / p)))


def pool_of(r):
    s = (r["family"], json.dumps(r["params"], sort_keys=True))
    return "training" if s in TRAIN else "held-out" if s in HELD else None


def pooled(rows, seeds):
    """{(pool, method, k, s, l, restarts): array of (gap, quantum_runs, pasqal_runs)} at n = 100 and 150."""
    out = defaultdict(list)
    for r in rows:
        if r["seed"] in seeds and r["n"] in (100, 150) and not r["exploded"] and pool_of(r):
            out[(pool_of(r), r["method"], r["k"], r["s"], r["l"], r["restarts"])].append(
                (r["gap_pct"], r["quantum_runs"], r["pasqal_runs"]))
    return {k: np.array(v, float) for k, v in out.items()}


def summary(a):
    lo, hi = bootstrap_ci(a[:, 0])
    return a[:, 0].mean(), lo, hi, a[:, 1].mean(), a[:, 2].mean()


# Fresh seeds 1020-1039: no pick, threshold or design choice ever read them.
fresh = load(files("fresh4c_*.jsonl", "fresh5b_s0.jsonl"))
FP = pooled(fresh, FRESH_SEEDS)
def _inst(r):
    return r["family"], json.dumps(r["params"], sort_keys=True), r["n"], r["seed"]


def rescore(rows, best):
    """Gap against the best size known for the instance from every record, not only these rows."""
    for r in rows:
        r["gap_pct"] = 100 * (best[_inst(r)] - r["size"]) / best[_inst(r)]
    return rows


RAW = {name: [json.loads(x) for x in open(EVAL / f"{name}.jsonl")]
       for name in ("mac5a", "emu5a", "uni5a", "fresh5b_greedy", *(f"fresh5b_s{i}" for i in range(5)))}
BEST = defaultdict(int)
for r in fresh + [r for rs in RAW.values() for r in rs]:
    if not r["exploded"]:
        BEST[_inst(r)] = max(BEST[_inst(r)], r["size"])
BANDIT_SEEDS = [pooled(rescore(RAW[f"fresh5b_s{i}"], BEST), FRESH_SEEDS) for i in range(5)]

CURVES = [  # label, method, configs, colour, marker, line style
    ("GLS", "gls", [(1, 1, 1), (2, 2, 5), (4, 2, 10), (8, 3, 30)], ORANGE, "o", "-"),
    ("Linear + value", L, [(1, 1, 1), (2, 2, 5), (4, 2, 10), (8, 3, 30)], MUTED, "s", "-"),
    ("Builder", B8, [(1, 1, 1), (2, 2, 5), (4, 2, 10), (8, 3, 30)], INK2, "^", "-"),
    ("Builder + search", S8, [(4, 2, b) for b in (6, 25, 50, 100, 200)], BLUE, "D", "-"),
]

print("\nFresh seeds, pooled: gap [95% CI], quantum runs, piece-solver runs")
for pool in ("training", "held-out"):
    for label, method, cfgs, *_ in CURVES:
        for c in cfgs:
            g, lo, hi, q, p = summary(FP[(pool, method, *c, 1)])
            print(f"  {pool:9s} {label:17s} {str(c):13s} {g:6.2f} [{lo:5.2f}, {hi:5.2f}] {q:7.1f} {p:6.1f}")
    for i, bp in enumerate(BANDIT_SEEDS):
        g, lo, hi, q, p = summary(bp[(pool, "bandit-r", 4, 2, 10, 1)])
        print(f"  {pool:9s} round bandit s{i}   {g:6.2f} [{lo:5.2f}, {hi:5.2f}] {q:7.1f} {p:6.1f}")
    g, lo, hi, *_ = summary(FP[(pool, "greedy-mindeg", None, None, None, 20)])
    print(f"  {pool:9s} greedy            {g:6.2f} [{lo:5.2f}, {hi:5.2f}]")


# Figure: gap against quantum runs on the fresh seeds.
fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.55), sharex=True)
for ax, pool, title, top in ((axes[0], "training", "(a) Families seen in training", 21),
                             (axes[1], "held-out", "(b) Held-out families", 9.5)):
    g, lo, hi, *_ = summary(FP[(pool, "greedy-mindeg", None, None, None, 20)])
    ax.axhspan(lo, hi, color=AQUA, alpha=0.12, lw=0)
    ax.axhline(g, color=AQUA, lw=1.4)
    ax.text(7.6, g - 0.15 * top / 9.5, "min-degree greedy, no quantum runs", color=INK2, fontsize=6.5, ha="left", va="top")
    for label, method, cfgs, col, mk, ls in CURVES:
        pts = np.array([summary(FP[(pool, method, *c, 1)]) for c in cfgs])
        ax.errorbar(pts[:, 3], pts[:, 0], yerr=[pts[:, 0] - pts[:, 1], pts[:, 2] - pts[:, 0]], color=col, lw=1.4,
                    ls=ls, marker=mk, ms=4.2 if mk != "D" else 3.8, mec="white", mew=0.6, elinewidth=0.6, capsize=0,
                    zorder=3, label=label)
    bp = np.array([summary(b[(pool, "bandit-r", 4, 2, 10, 1)]) for b in BANDIT_SEEDS])
    ax.scatter(bp[:, 3], bp[:, 0], marker="v", s=22, color=BLUE, facecolor="white", lw=1.0, zorder=4,
               label="Round bandit (5 seeds)")
    ax.set_xscale("log")
    ax.set_xlim(7, 2.2e3)
    ax.set_ylim(0, top)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Mean quantum runs per instance")
axes[0].set_ylabel("Mean gap to the best known (%)")
h, lab = axes[0].get_legend_handles_labels()
order = [1, 2, 3, 4, 0]
fig.legend([h[i] for i in order], [lab[i] for i in order], loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.07),
           handlelength=1.8, columnspacing=1.4)
fig.tight_layout(w_pad=1.5)
fig.savefig(OUT / "tradeoff.png", dpi=220)
save(fig, "tradeoff.pdf")


# Figure: per-setting gaps on the fresh seeds, at each method's headline configuration.
SETTING_LABEL = {
    ("er", '{"p": 0.1}'): "ER $p=0.1$", ("er", '{"p": 0.25}'): "ER $p=0.25$", ("er", '{"p": 0.75}'): "ER $p=0.75$",
    ("power_law", '{"m": 2}'): "BA $m=2$", ("power_law", '{"m": 8}'): "BA $m=8$",
    ("sbm", '{"blocks": 4, "p_in": 0.3, "p_out": 0.03}'): "SBM",
    ("factor", '{"theta": 0.2}'): r"Factor $\theta=0.2$", ("factor", '{"theta": 0.3}'): r"Factor $\theta=0.3$",
    ("factor", '{"market": 1.0, "theta": 0.3}'): r"Crisis $\theta=0.3$",
    ("factor", '{"market": 1.0, "theta": 0.4}'): r"Crisis $\theta=0.4$",
    ("factor", '{"market": 1.0, "theta": 0.5}'): r"Crisis $\theta=0.5$",
    ("unit_disk", '{"radius": 0.1}'): "Unit disk $r=0.10$", ("unit_disk", '{"radius": 0.15}'): "Unit disk $r=0.15$",
}
COLS = [("GLS", "gls", (4, 2, 10, 1)), ("Linear\n+ value", L, (4, 2, 10, 1)), ("DAgger\nGNN", D, (4, 2, 10, 1)),
        ("Builder", B8, (4, 2, 10, 1)), ("Builder\n+ search", S8, (4, 2, 100, 1)),
        ("Round\nbandit", "bandit-r", (4, 2, 10, 1)), ("Greedy", "greedy-mindeg", (None, None, None, 20))]
settings = [(f, json.dumps(p, sort_keys=True)) for f, p in TRAIN_SETTINGS + HELD_OUT_SETTINGS]
cell = defaultdict(list)
for r in fresh:
    if r["seed"] in FRESH_SEEDS and r["n"] in (100, 150) and not r["exploded"]:
        cell[(r["family"], json.dumps(r["params"], sort_keys=True), r["method"], r["k"], r["s"], r["l"],
              r["restarts"])].append(r["gap_pct"])
G = np.array([[np.mean(cell[(*s, m, *c)]) for _, m, c in COLS] for s in settings])
fig, ax = plt.subplots(figsize=(COL_W, 3.45))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
im = ax.imshow(np.sqrt(G), cmap=cmap, vmin=0, vmax=np.sqrt(20), aspect="auto")
for i in range(G.shape[0]):
    best = min(round(v, 1) for v in G[i, 1:6])
    for j in range(G.shape[1]):
        v = G[i, j]
        txt = f"{v:.1f}"
        weight = "bold" if 1 <= j <= 5 and round(v, 1) == best else "normal"
        ax.text(j, i, txt, ha="center", va="center", fontsize=6.2, weight=weight,
                color="white" if v > 6 else INK)
ax.set_xticks(range(len(COLS)), [c[0] for c in COLS], fontsize=6.2)
ax.set_yticks(range(len(settings)), [SETTING_LABEL[s] for s in settings], fontsize=6.5)
ax.xaxis.tick_top()
ax.tick_params(length=0)
ax.grid(False)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.axhline(len(TRAIN_SETTINGS) - 0.5, color=INK, lw=0.8)
ax.axvline(0.5, color="white", lw=2)
ax.axvline(5.5, color="white", lw=2)
ax.text(len(COLS) - 0.4, (len(TRAIN_SETTINGS) - 1) / 2, "training", rotation=-90, ha="left", va="center", fontsize=6.5,
        color=INK2)
ax.text(len(COLS) - 0.4, len(TRAIN_SETTINGS) + (len(HELD_OUT_SETTINGS) - 1) / 2, "held-out", rotation=-90, ha="left",
        va="center", fontsize=6.5, color=INK2)
save(fig, "families.pdf")


# Figure: whole decompositions on the emulator, exact solver and uniform control (six fresh blocks).
emu = {name: rescore(RAW[name], BEST) for name in ("mac5a", "emu5a", "uni5a")}
EMU_METHODS = [("Linear + value", L, (4, 2, 10)), ("Builder", B8, (4, 2, 10)),
               ("Builder + search", S8, (4, 2, 100)), ("Round bandit", "bandit-r", (4, 2, 10)),
               ("Fixed arm", "bandit-r", (4, 2, 9002))]
greedy_emu = pooled(rescore(RAW["fresh5b_greedy"], BEST), range(1020, 1026))
print("\nEmulator blocks (seeds 1020-1025), pooled gap, quantum runs, piece-solver runs")
fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.15), sharey=True)
SOLVERS = [("mac5a", "Exact solver", INK2, "o", INK2), ("emu5a", "EMU_MPS", BLUE, "D", BLUE),
           ("uni5a", "Uniform + repair", "white", "o", MUTED)]
for ax, pool, title in ((axes[0], "training", "(a) Families seen in training"),
                        (axes[1], "held-out", "(b) Held-out families")):
    gg = greedy_emu[(pool, "greedy-mindeg", None, None, None, 20)][:, 0].mean()
    ax.axvline(gg, color=AQUA, lw=1.4)
    for j, (name, lab, fc, mk, ec) in enumerate(SOLVERS):
        P = pooled(emu[name], range(1020, 1026))
        for i, (ml, m, c) in enumerate(EMU_METHODS):
            a = P[(pool, m, *c, 1)]
            g, lo, hi, q, p = summary(a)
            if pool == "training" or True:
                print(f"  {pool:9s} {name} {ml:17s} {g:6.2f} [{lo:5.2f}, {hi:5.2f}] {q:6.1f} {p:6.1f} n={len(a)}")
            y = i + (j - 1) * 0.22
            ax.plot([lo, hi], [y, y], color=ec, lw=0.7, zorder=2)
            ax.scatter([g], [y], marker=mk, s=16, facecolor=fc, edgecolor=ec, lw=0.9, zorder=3,
                       label=lab if i == 0 else None)
    ax.set_xlim(0, 5.8)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Mean gap to the best known (%)")
    ax.grid(axis="y", visible=False)
    ax.text(gg + 0.06, -0.72, "greedy", color=INK2, fontsize=6.5, va="center", ha="left")
axes[0].set_yticks(range(len(EMU_METHODS)), [m[0] for m in EMU_METHODS])
axes[0].set_ylim(len(EMU_METHODS) - 0.5, -0.95)
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.08))
fig.tight_layout(w_pad=1.2)
save(fig, "emulator.pdf")
print(f"  greedy: training {greedy_emu[('training', 'greedy-mindeg', None, None, None, 20)][:, 0].mean():.2f}, "
      f"held-out {greedy_emu[('held-out', 'greedy-mindeg', None, None, None, 20)][:, 0].mean():.2f}")




# Figure: one GLS piece, in the problem graph and on the 61-trap lattice.
import random
import networkx as nx
from learned_selection.growth import code_start_trap, gls_score, grow_piece, lattice_adjacency
from learned_selection.instances import erdos_renyi
from learned_selection.lattice import is_valid_piece, lattice_graph, load_coords
from learned_selection.solvers import exact_mis

coords = load_coords()
adj, avg = lattice_adjacency(coords)
Lg = lattice_graph(coords)
g = erdos_renyi(40, 0.1, seed=7)
pieces = [grow_piece(g, v, adj, code_start_trap(len(adj)), random.Random(0), gls_score(avg)(g)) for v in g]
piece = max(pieces, key=len)
assert is_valid_piece(g, piece, Lg)
sub = g.subgraph(piece)
mis = set(exact_mis(sub, workers=1)[0])
removed = set(mis) | {u for v in mis for u in g[v]}
print(f"\nExample piece: {len(piece)} of {len(g)} vertices, MIS of the piece {len(mis)}, "
      f"removed with its neighbourhood {len(removed)}")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(PAGE_W, 2.55), gridspec_kw={"width_ratios": [1, 1]})
pos = nx.kamada_kawai_layout(g)
for u, v in g.edges():
    inside = u in piece and v in piece
    a1.plot(*zip(pos[u], pos[v]), color=BLUE if inside else PALE, lw=1.0 if inside else 0.6, zorder=1)
for v in g:
    if v in mis:
        a1.scatter(*pos[v], s=34, color="#0d366b", edgecolor="white", lw=0.8, zorder=3)
    elif v in piece:
        a1.scatter(*pos[v], s=30, color=BLUE, edgecolor="white", lw=0.8, zorder=3)
    else:
        a1.scatter(*pos[v], s=22, color="white", edgecolor=MUTED, lw=0.8, zorder=2)
a1.set_title("(a) Problem graph and one GLS piece", loc="left")
for t, (x, y) in coords.items():
    a2.scatter(x, y, s=26, facecolor="white", edgecolor=PALE, lw=0.8, zorder=1)
for u, v in sub.edges():
    a2.plot(*zip(coords[piece[u]], coords[piece[v]]), color=BLUE, lw=1.0, zorder=2)
for v, t in piece.items():
    a2.scatter(*coords[t], s=40, color="#0d366b" if v in mis else BLUE, edgecolor="white", lw=0.8, zorder=3)
st = coords[code_start_trap(len(adj))]
a2.scatter(*st, s=95, facecolor="none", edgecolor=ORANGE, lw=1.1, zorder=4)
a2.set_title("(b) The piece on the 61-trap triangular layout", loc="left")
for ax in (a1, a2):
    ax.set_aspect("equal")
    ax.axis("off")
a2.plot([13, 18], [-20.5, -20.5], color=INK2, lw=1)
a2.text(15.5, -21.3, r"5 $\mu$m", ha="center", va="top", fontsize=6.5, color=INK2)
from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="o", ls="", color=BLUE, mec="white", ms=5.5, label="vertex in the piece"),
           Line2D([], [], marker="o", ls="", color="#0d366b", mec="white", ms=5.5, label="maximum independent set of the piece"),
           Line2D([], [], marker="o", ls="", mfc="none", mec=ORANGE, ms=8, label="seed trap"),
           Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, ms=5, label="other vertex or free trap")]
fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout()
save(fig, "piece.pdf")
