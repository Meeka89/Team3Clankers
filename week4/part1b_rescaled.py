"""
By Carter Owens
"""


def normalize(channel):
    # returns a list


    # finds the largest value in channel
    biggest = max(channel)


    # returns the channel back but normalized by dividing
    # every input by the biggest value
    return [reading / biggest for reading in channel]


blade_angle = [8.5, 9.5, 9.9, 9.0] # degrees off-vertical
balance = [0.65, 0.80, 0.80, 0.90] # stance reading, [0, 1]
breath = [1.2, 1.3, 0.5, 1.0] # exhalations per second
clean = [1, 1, 0, 1] # ground truth: clean strike?
sensings = [[blade_angle[i],balance[i], breath[i]] for i in range(4)]




# normalize each "channel"
blade_angle_norm = normalize(blade_angle)
balance_norm = normalize(balance)
breath_norm = normalize(breath)


# normalized sensings
sensings_norm = [[blade_angle_norm[i],balance_norm[i], breath_norm[i]] for i in range(4)]


print("-- Sensing 0 Normalized --")
for value in sensings_norm[0]:
    print(f"{value:.4f}")


from part1_multi_input import gradient_descent_multi


weights = [0.1, 0.2, -0.1]
true = 1
alpha_sc = 0.1
alpha_raw = 0.01
iterations = 20


_, error_history_raw, _ = gradient_descent_multi(sensings[0],
                                                 weights,                                                                      
                                                 true,
                                                 alpha_raw,
                                                 iterations)


_, error_history_norm, _ = gradient_descent_multi(sensings_norm[0],
                                                 weights,                                                                      
                                                 true,
                                                 alpha_sc,
                                                 iterations)


print("")
for it in range(iterations):
    print(f"Iteration {it}: Raw Error: {error_history_raw[it]:.6f}, Scaled Error: {error_history_norm[it]:.6f}")




"""
Comment Section


a) Raw inputs at alpha = 0.1 diverge; scaled inputs at the same alpha do not. Explain it in
terms of weight_delta = delta * input.


The size of the weight update is scaled by the input because weight delta is alpha * delta * input.
THis means that for the raw input with the large input and a larger input cause it to overshoot its
prediction and it will continue overshooting, causing divergence. As for scaled data data, the same alpha
gives a controlled step because every input value is less than, causing it to not diverge.


(b) Your error after scaling is larger than the raw alpha = 0.01 run at every iteration, not just
the first. Why is that not a regression? What are you actually comparing when you say one
run is better than the other?


The runs are not necessarily the same. The raw run looks better because it happened to be a good fit for that
scale, while the scaled run's alpha is a generic value that is not tuned. If the numbers were to change on the
raw run, it is highly likely that it would not do well. Compared to the scaled run, you could change the alpha
and it would do better because the input stays in the small range.


(c) Week 2 also gave you min-max. Applied to blade_angle, what does it make sensing 0's first
entry? What then happens to that weight, and which later part of this assignment does that
resemble?


Sensing 0's first entry would be 0. So then when it tries to udpate the weight, it simply doesn't and
does not move. This resembles dead neurons or freezing weights.


"""


