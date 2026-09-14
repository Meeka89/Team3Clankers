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