from micrograd.module import Value

xs = [[2, 3, -1], [3, -1, 0.5], [0.5, 1, 1], [1, 1, -1]]
ys = [1.0, -1.0, -1.0, 1.0]
ys_val = [Value(y) for y in ys]