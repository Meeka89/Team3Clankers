def w_sum(a, b):
    assert len(a) == len(b)

    output = 0

    for i in range(len(a)):
        output += a[i] * b[i]

    return output


def vect_mat_mul(vect, matrix):
    # Number of inputs must match number of columns in matrix
    assert len(vect) == len(matrix[0])

    output = []

    for i in range(len(matrix)):
        output.append(w_sum(vect, matrix[i]))

    return output
