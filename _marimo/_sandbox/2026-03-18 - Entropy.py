import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    p = np.linspace(1e-08, 1 - 1e-08, 500)
    # Probability values (excluding 0 and 1 to avoid log(0))
    q = 1 - p
    _entropy = -p * np.log2(p) - q * np.log2(q)
    _negative_quadratic_entropy = -(p ** 2 + q ** 2)
    # Entropy H(p) = -p*log2(p) - q*log2(q)
    plt.figure(figsize=(8, 5))
    plt.plot(p, _entropy, label='Shannon entropy')
    # Negative quadratic entropy: - (p^2 + q^2)
    plt.plot(p, _negative_quadratic_entropy, label='Negative quadratic entropy')
    plt.xlabel('Probability $p$')
    plt.ylabel('Value')
    plt.title('Entropy vs Negative Quadratic Entropy for Two-Outcome Probability')
    plt.legend()
    plt.grid(True)
    plt.show()
    return np, plt


@app.cell
def _(np, plt):
    from mpl_toolkits.mplot3d import Axes3D
    num_points = 200
    x = np.linspace(1e-08, 1 - 1e-08, num_points)
    y = np.linspace(1e-08, 1 - 1e-08, num_points)
    X, Y = np.meshgrid(x, y)
    Z = 1 - X - Y
    mask = Z > 1e-08
    X_valid = X[mask]
    Y_valid = Y[mask]
    Z_valid = Z[mask]
    _entropy = -(X_valid * np.log2(X_valid) + Y_valid * np.log2(Y_valid) + Z_valid * np.log2(Z_valid))
    _negative_quadratic_entropy = -(X_valid ** 2 + Y_valid ** 2 + Z_valid ** 2)
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_trisurf(X_valid, Y_valid, _entropy, cmap='viridis', linewidth=0.1)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.set_zlabel('Shannon Entropy')
    ax1.set_title('Shannon Entropy for x + y + z = 1')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_trisurf(X_valid, Y_valid, _negative_quadratic_entropy, cmap='plasma', linewidth=0.1)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')
    ax2.set_zlabel('Negative Quadratic Entropy')
    ax2.set_title('Negative Quadratic Entropy for x + y + z = 1')
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
