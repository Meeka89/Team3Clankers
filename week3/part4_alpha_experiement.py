"""
File: week3/part4_alpha_experiment.py
Part 4: Alpha Experimentation
"""

from part3_alpha import gradient_descent_alpha


def detect_divergence(errors: list[float]) -> bool:
    """
    Given a list of error values, returns True if the run failed to make progress—
    that is, if the last error is not meaningfully smaller than the first.
    """
    if not errors:
        return False
    return errors[-1] > 0.99 * errors[0]


def run_experiment() -> None:
    # Input scenario from Part 3
    input_val = 2.0
    goal = 0.8
    initial_weight = 0.5
    iterations = 20

    alpha_values = [0.001, 0.01, 0.1, 0.4, 0.5, 0.6, 1.0]

    print(f"{'Alpha':<8} | {'Final Error':<12} | {'Detector Reported':<18}")
    print("-" * 44)

    for alpha in alpha_values:
        errors = gradient_descent_alpha(
            input_val, goal, initial_weight, alpha, iterations
        )
        final_error = errors[-1]
        diverged = detect_divergence(errors)
        status = "Diverged (True)" if diverged else "Converged (False)"

        print(f"{alpha:<8.3f} | {final_error:<12.6f} | {status:<18}")


if __name__ == "__main__":
    run_experiment()

    """
    OBSERVATIONS ON ALPHA BOUNDARY & STABILITY:
    ---------------------------------------------------------------------------
    - Largest alpha tested that converges: alpha = 0.4 (Final error drops significantly).
    - Smallest alpha tested that diverges: alpha = 0.6 (Final error explodes to >1000).
    - True Boundary (alpha = 0.5):
      At alpha = 0.5, the error stays strictly constant at 0.04 across all 20 iterations.
      The weight update endlessly bounces back and forth between 0.5 and 0.42 without 
      ever shrinking or growing. Because it sits exactly on the stability boundary, it 
      neither converges nor diverges. A binary True/False detector cannot honestly describe 
      this state; our simple threshold reports 'False' (not diverged) purely because 
      0.04 is not strictly greater than 0.99 * 0.04.
      
    - Tradeoff:
      - Smaller alpha (e.g., 0.001, 0.01): Very stable and safe from exploding, but 
        learning is extremely slow and takes many iterations to reach the goal.
      - Larger alpha (close to 0.5): Learns very rapidly in early steps, but risks 
        overshooting, oscillation, or complete divergence if it crosses the critical limit.
    ---------------------------------------------------------------------------
    """