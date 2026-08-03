"""
Data: held-out decomposition of c = exp(-d/kappa) * (1 - sigma) across
MAPPO training checkpoints (n=5, each checkpoint at eval seed 10000+u,
baseline at 10000). Source: thesis Table tab:eval-decomp.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

labels = ["baseline", "u4", "u9", "u14", "u19", "u24"]
x = range(len(labels))

c          = [0.158, 0.131, 0.139, 0.155, 0.119, 0.141]
similarity = [0.555, 0.564, 0.572, 0.567, 0.558, 0.571]
unsafety   = [0.276, 0.222, 0.237, 0.269, 0.207, 0.240]

fig, ax = plt.subplots(figsize=(7.5, 4.2))

ax.plot(x, similarity, "o-", lw=2.2, ms=6, color="#c44e52",
        label=r"similarity  $\exp(-d/\kappa)$")
ax.plot(x, unsafety, "s-", lw=2.2, ms=6, color="#4c72b0",
        label=r"unsafety  $(1-\sigma)$")
ax.plot(x, c, "^--", lw=2.2, ms=6, color="#55a868",
        label=r"consensus cost  $c$")

ax.annotate("latent term flat throughout training\n(0.555 → 0.571, trending up)",
            xy=(3, 0.567), xytext=(1.15, 0.66),
            fontsize=9, color="#c44e52",
            arrowprops=dict(arrowstyle="->", color="#c44e52", lw=1.2))

ax.annotate("$c$ tracks the Judge term",
            xy=(4, 0.207), xytext=(4.05, 0.30),
            fontsize=9, color="#4c72b0",
            arrowprops=dict(arrowstyle="->", color="#4c72b0", lw=1.2))

ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_xlabel("training checkpoint")
ax.set_ylabel("value")
ax.set_ylim(0.05, 0.72)
ax.set_title("The latent-similarity term contributes no training signal",
             fontsize=12, pad=12)
ax.legend(frameon=False, loc="lower left", fontsize=9)
ax.grid(axis="y", alpha=0.25, lw=0.7)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig("/home/claude/mechanism_decomposition.png", dpi=200)
print("done")
