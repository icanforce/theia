
def find_modules(x_name, y_name, U, V, th):
    assert U.shape[1] == V.shape[1]
    K = U.shape[1]

    def f(name, M, k):
        return [name[i] for i in range(M.shape[0]) if M[i][k] > th]

    return [(f(x_name, U, k), f(y_name, V, k)) for k in range(K)]
