"""
Visualize the 2-D Clenshaw-Curtis Smolyak sparse grid over (a_g, b_g),
contrasted with the full tensor grid, to make 'sparse' intuitive.

Matches the paper's construction: nested CC rule, level l -> exact to degree 2^l - 1.
While the full tensor grid scales as O(n^d), the Smolyak grid scales as
O(2^l * l^(d-1)), mitigating the curse of dimensionality.

    python make_smolyak_fig.py            # default LEVEL below
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
#  Grid level.  LEVEL = 5 -> clean 2-D slice (145 sparse / 1089 tensor).
#  LEVEL = 4 matches the 11-D level-4 rule used in the paper (m_SG=12,497).
# ----------------------------------------------------------------------
LEVEL = 5

# ---- 1-D Clenshaw-Curtis nested nodes at "level" i ----
# level 0 -> 1 node (0); level i>=1 -> 2^i + 1 nodes on [-1,1], nested.
def cc_nodes(i):
    if i == 0:
        return np.array([0.0])
    m = 2**i + 1
    j = np.arange(m)
    x = -np.cos(np.pi * j / (m - 1))          # CC nodes on [-1,1]
    x[np.abs(x) < 1e-12] = 0.0
    return np.unique(np.round(x, 12))

# ---- 2-D Smolyak sparse grid of level L ----
# union over i1 + i2 <= L of tensor(cc_nodes(i1), cc_nodes(i2))
def smolyak_2d(L):
    pts = set()
    for i1 in range(L + 1):
        for i2 in range(L + 1):
            if i1 + i2 <= L:
                X = cc_nodes(i1); Y = cc_nodes(i2)
                for a in X:
                    for b in Y:
                        pts.add((round(a, 12), round(b, 12)))
    return np.array(sorted(pts))

# ---- full tensor grid at the same 1-D resolution (level L in each dim) ----
def tensor_2d(L):
    X = cc_nodes(L)
    A, B = np.meshgrid(X, X)
    return np.column_stack([A.ravel(), B.ravel()])

# ---- physical bounds + truth for (a_g, b_g) (graupel, Derek Table 1) ----
AG_MIN, AG_MAX = 50.0, 1200.0
BG_MIN, BG_MAX = 0.10, 0.90
AG_TRUE, BG_TRUE = 400.0, 0.40

def to_phys(P):
    ag = AG_MIN + (P[:, 0] + 1) / 2 * (AG_MAX - AG_MIN)
    bg = BG_MIN + (P[:, 1] + 1) / 2 * (BG_MAX - BG_MIN)
    return ag, bg

sg = smolyak_2d(LEVEL)
tg = tensor_2d(LEVEL)
print(f"level L={LEVEL}:  Smolyak sparse = {len(sg)} points,  "
      f"full tensor = {len(tg)} points")
print(f"reduction factor = {len(tg) / len(sg):.1f}x")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# (a) full tensor grid
ag_t, bg_t = to_phys(tg)
axes[0].scatter(ag_t, bg_t, s=12, c="0.55", edgecolors="none")
axes[0].set_title(f"(a) Full tensor grid  ({len(tg)} points)", fontsize=11)

# (b) Smolyak sparse grid
ag_s, bg_s = to_phys(sg)
axes[1].scatter(ag_s, bg_s, s=20, c="tab:blue", edgecolors="white", linewidths=0.4)
axes[1].set_title(f"(b) Clenshaw--Curtis Smolyak grid, level {LEVEL}  "
                  f"({len(sg)} points)", fontsize=11)

for ax in axes:
    ax.axvline(AG_TRUE, ls=":", c="0.4", lw=0.9)
    ax.axhline(BG_TRUE, ls=":", c="0.4", lw=0.9)
    ax.plot(AG_TRUE, BG_TRUE, "x", c="red", ms=9, mew=2, zorder=5)
    ax.set_xlabel(r"$a_g$")
    ax.set_ylabel(r"$b_g$")
    ax.set_xlim(AG_MIN, AG_MAX)
    ax.set_ylim(BG_MIN, BG_MAX)
    ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig("smolyak_grid_ag_bg.png", dpi=200, bbox_inches="tight")
fig.savefig("smolyak_grid_ag_bg.pdf", bbox_inches="tight")
print("saved smolyak_grid_ag_bg.png / .pdf")
