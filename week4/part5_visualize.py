#-----part5_visualize.py-----#
#----------Andrew_Martin------------#

import matplotlib  # type: ignore[reportMissingModuleSource]
matplotlib.use("Agg")  # render to a file; no display needed
import matplotlib.pyplot as plt  # type: ignore[reportMissingModuleSource]

from part1_multi_input import gradient_descent_multi
from part4_freeze import gradient_descent_frozen

blade_angle, balance, breath = 8.5, 0.65, 1.2
input_val = [blade_angle, balance, breath]
start_weights = [0.1, 0.2, -0.1]
true = 1.0
alpha = 0.01
iterations = 10


# --- Figure 1: Part 1's weights over time ---
weights, errors, weight_history = gradient_descent_multi(
    list(input_val), list(start_weights), true, alpha, iterations
)

for j, name in enumerate(["blade_angle", "balance", "breath"]):
    plt.plot([w[j] for w in weight_history], label=name)
plt.xlabel("iteration")
plt.ylabel("weight value")
plt.title("Part 1 weights, alpha = 0.01")
plt.legend()
plt.savefig("fig1_weights.png", dpi=150)
plt.close()  # start a clean figure for the next plot


# --- Figure 2: frozen vs. unfrozen ---
_, _, balance_frozen_hist = gradient_descent_frozen(
    list(input_val), list(start_weights), true, alpha, iterations, frozen=[0, 2]
)
_, _, breath_frozen_hist = gradient_descent_frozen(
    list(input_val), list(start_weights), true, alpha, iterations, frozen=[0, 1]
)

plt.plot([w[1] for w in balance_frozen_hist], label="balance, frozen=[0, 2], alpha=0.01")
plt.plot([w[1] for w in weight_history], label="balance, unfrozen, alpha=0.01")
plt.plot([w[2] for w in breath_frozen_hist], label="breath, frozen=[0, 1], alpha=0.01")
plt.plot([w[2] for w in weight_history], label="breath, unfrozen, alpha=0.01")
plt.xlabel("iteration")
plt.ylabel("weight value")
plt.title("Part 4 frozen vs. unfrozen weight trajectories")
plt.legend()
plt.savefig("fig2_frozen.png", dpi=150)
plt.close()

print("Saved fig1_weights.png and fig2_frozen.png")

"""
5
In Figure 2, freezing the weights beside a free weight makes that free
weight move further and faster than it does in the unfrozen baseline: over
10 iterations, frozen balance moves +0.00893 vs. only +0.00123 unfrozen,
and frozen breath moves +0.01575 vs. only +0.00227 unfrozen -- roughly 7x
more in both cases. In the unfrozen run, blade_angle (the largest input)
does most of the work closing the gap between prediction and target, so
delta shrinks quickly and every weight's per-step update shrinks with it.
With blade_angle frozen, it can't help close that gap, so delta stays
larger for longer and whichever single weight is still free has to absorb
all of that lingering correction alone.

blade_angle is the steepest line in Figure 1 (0.1 -> 0.116, a change of
about +0.016), far more than balance (+0.001) or breath (+0.002). That
matches Part 1's own comment block, which predicted blade_angle would
change the most because its input (8.5) is so much larger than balance's
(0.65) or breath's (1.2) -- weight_delta = delta * input, so the same
shared delta gets scaled up far more for blade_angle.

The balance and breath lines in Figure 1 are NOT actually flat -- they do
move (by +0.001 and +0.002 respectively) -- they only look flat because
the y-axis has to span blade_angle's much larger swing, which squashes
their movement down to a sliver. To see their real shape, plot balance and
breath on their own separate axes (e.g. one subplot per weight, or a
zoomed-in plot restricted to their own value range) instead of sharing one
y-axis with blade_angle.
"""
