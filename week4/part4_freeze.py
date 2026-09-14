def ele_mul(scalar, vector):
    return [scalar * x for x in vector]


def gradient_descent_frozen(input_vec, weights, true, alpha, iterations, frozen):
    weights = list(weights)
    error_history = []
    weight_history = [list(weights)]

    for _ in range(iterations):
        pred = sum(i * w for i, w in zip(input_vec, weights))
        error = (pred - true) ** 2
        delta = pred - true
        error_history.append(error)

        raw_weight_deltas = ele_mul(delta, input_vec)

        # Freeze specified weights by setting delta to 0
        weight_deltas = [
            0.0 if i in frozen else raw_weight_deltas[i]
            for i in range(len(weights))
        ]

        # Update unfrozen weights
        weights = [
            w - alpha * wd for w, wd in zip(weights, weight_deltas)
        ]
        weight_history.append(list(weights))

    return weights, error_history, weight_history


if __name__ == "__main__":
    sensing_0 = [8.5, 0.65, 1.2]
    init_weights = [0.1, 0.2, -0.1]
    target = 1.0
    num_iters = 5

    configs = [
        {"name": "Baseline (Unfrozen)", "frozen": [], "alpha": 0.01},
        {"name": "Only Balance Learns", "frozen": [0, 2], "alpha": 0.3},
        {"name": "Only Breath Learns", "frozen": [0, 1], "alpha": 0.3},
    ]

    for cfg in configs:
        print(f"=== {cfg['name']} (frozen={cfg['frozen']}, alpha={cfg['alpha']}) ===")
        final_w, errs, w_hist = gradient_descent_frozen(
            sensing_0, init_weights, target, cfg["alpha"], num_iters, cfg["frozen"]
        )

        for it in range(num_iters):
            pred = sum(i * w for i, w in zip(sensing_0, w_hist[it]))
            print(
                f"Iter {it}: Pred = {pred:.4f}, Error = {errs[it]:.6f}, "
                f"Weights = {[round(w, 4) for w in w_hist[it]]}"
            )
        print(f"Final Weights: {[round(w, 4) for w in final_w]}\n")

"""
COMMENT BLOCK: PART 4 - FREEZING ANALYSIS

1. Identification of Compensating Weights & Absorbing Slack:
   - Baseline (frozen=[]): All three weights update concurrently based on their input 
     magnitudes.
   - Configuration 1 (frozen=[0, 2]): The 'balance' weight (index 1) does ALL the correcting. 
     Because blade_angle and breath are locked and silent, the balance weight is forced 
     to absorb all the residual error ("slack") to drive the prediction toward 1.0.
   - Configuration 2 (frozen=[0, 1]): The 'breath' weight (index 2) does ALL the correcting 
     for the same reason.

2. Mathematical Explanation of Learning-Rate Tolerance:
   Each iteration multiplies the prediction miss (delta) by the factor (1 - alpha * S), 
   where S = sum(input[i]^2) evaluated ONLY over active (unfrozen) inputs.

   - Baseline (All free, alpha = 0.01):
     S = 8.5^2 + 0.65^2 + 1.2^2 = 72.25 + 0.4225 + 1.44 = 74.1125
     Scaling factor = 1 - (0.01 * 74.1125) = 1 - 0.741125 = 0.258875
     (The miss shrinks stably by ~74% per step).
     *Note: If alpha=0.3 were used here, 1 - (0.3 * 74.1125) = -21.23, causing extreme 
     divergence and flipping signs.*

   - Balance Only Free (frozen=[0, 2], alpha = 0.3):
     S = 0.65^2 = 0.4225
     Scaling factor = 1 - (0.3 * 0.4225) = 1 - 0.12675 = 0.87325
     (The miss shrinks stably by ~12.7% per step).

   - Breath Only Free (frozen=[0, 1], alpha = 0.3):
     S = 1.2^2 = 1.44
     Scaling factor = 1 - (0.3 * 1.44) = 1 - 0.432 = 0.568
     (The miss shrinks stably by ~43.2% per step).

   Conclusion: Freezing index 0 removes the massive blade_angle input (8.5) from S, 
   dropping S from 74.1125 to <= 1.44. Because the stability limit requires |1 - alpha * S| < 1, 
   removing the large input allows us to safely scale alpha up by 30x without overshooting or exploding.
"""
