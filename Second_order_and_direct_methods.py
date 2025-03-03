import numpy as np
import matplotlib.pyplot as plt

# Nastaveni PP
A = 1
B = 5


ITERATIONS = 100
MAX_CALLS = 1000

START_X, START_Y = -1, -1

STOP_CRITERION = 1e-2

DELTA = 1e-4
EPS_CYCLIC = 1e-2
EPS_HOOKE_JEEVES = 1e-6
ALPHA_HOOKE_JEEVES = 1

SIMPLEX_S = np.array([
    [-1, -0.8, -0.5],
    [-1, -0.8, -1]
])




# Volba metody, pro conjugate descent = False
GRADIENT_DESCENT_METHOD = 1
OPTIMAL_ALPHA = 1

# Rosenbrockova funkce + gradient
f = lambda x, y: (A - x)**2 + B * (y - x**2)**2

df = lambda x, y: np.array([-2 * (A - x) - 4 * B * x * (y - x**2), 2 * B * (y - x**2)])

class DFP:
    def __init__(self, m):
        self.Q = np.eye(m)

    def init(self, f, grad_f, x):
        self.m = len(x)
        self.Q = np.eye(self.m)  # Inicializace identity matice

    def step(self, f, grad_f, x):
        g = grad_f(x[0], x[1])  # Gradient
        d = -np.dot(self.Q, g)  # Směr klesání
        alpha = line_search(f, x, d)  # Hledání optimálního kroku
        x_new = x + alpha * d  # Nový bod

        g_new = grad_f(x_new[0], x_new[1])  # Nový gradient
        delta_x = x_new - x
        delta_g = g_new - g

        # Aktualizace matice Q podle DFP metody
        self.Q = self.Q - np.outer(np.dot(self.Q, delta_g), np.dot(self.Q, delta_g)) / np.dot(delta_g, np.dot(self.Q, delta_g)) \
                    + np.outer(delta_x, delta_x) / np.dot(delta_x, delta_g)
        
        return x_new


def fibonacci_search(f, a, b, n=50, eps=0.01):
    fi = (1 + np.sqrt(5)) / 2
    s = (1 - np.sqrt(5)) / (1 + np.sqrt(5))
    ro = (1 - s**n) / (fi * (1 - s**(n + 1)))
    d = ro * b + (1 - ro) * a
    yd = f(d)

    for i in range(1, n):
        if i == n - 1:
            c = eps * a + (1 - eps) * d
        else:
            c = ro * a + (1 - ro) * b
        
        yc = f(c)
        
        if yc < yd:
            b, d, yd = d, c, yc
        else:
            a, b = b, c
        
    return (a + b) / 2

# Vypocet optimalni alfy
def line_search(f, x, d):
    f_line = lambda alpha: f(x[0] + alpha * d[0], x[1] + alpha * d[1])
    a, b = 0, 2   # Počáteční interval, nastavitelne
    alpha_opt = fibonacci_search(f_line, a, b)
    return alpha_opt

def cyclic_coordinate_descent_with_acceleration_step(f, x0, y0, eps, max_calls=MAX_CALLS, max_iter = ITERATIONS):
    x = np.array([x0, y0])
    n = len(x)
    delta = np.inf
    history = []
    function_calls = 0
    iter = 0
    
    while abs(delta) > eps and function_calls < max_calls and iter < max_iter:
        x_prev = x.copy()
        
        for i in range(n):
            if function_calls >= max_calls:
                break
            d = np.zeros(n)
            d[i] = 1  # Jednosměrný základní vektor
            
            # Hledání optimálního kroku podél souřadnice
            alpha = line_search(f, x, d)
            x = x + alpha * d
            history.append((x[0], x[1], f(x[0], x[1])))
            function_calls += 1
        
        if function_calls >= max_calls:
            break
        
        # Acceleration step
        d = x - x_prev
        alpha = line_search(f, x, d)
        x = x + alpha * d
        history.append((x[0], x[1], f(x[0], x[1])))

        function_calls += 1

        iter += 1
        
        delta = np.linalg.norm(x - x_prev)
    print(f"Minimum cyclic coordinate descent: x= {x[0]}, y ={x[1]}, f(x,y) = {f(x[0], x[1])}")
    return history

def hooke_jeeves(f, x0, y0, alpha, eps, gamma = 0.5, max_calls = MAX_CALLS, max_iter = ITERATIONS):
    x = np.array([x0, y0])
    y = f(x[0], x[1])
    n = len(x)
    history = []
    function_calls = 0
    iter = 0

    while alpha > eps and function_calls < max_calls and iter < max_iter:
        improved = False
        x_best, y_best = x, y
        for i in range(n):
            for sgn in (-1, 1):
                x_prime = x + sgn * alpha * np.eye(n)[i]
                y_prime = f(x_prime[0], x_prime[1])
                function_calls += 1
                if y_prime < y_best:
                    x_best, y_best = x_prime, y_prime
                    improved = True
        x, y = x_best, y_best
        if not improved:
            alpha *= gamma
        history.append((x[0], x[1], f(x[0], x[1])))
        iter += 1
    print(f"Minimum hooke jeeves: x= {x[0]}, y ={x[1]}, f(x,y) = {f(x[0], x[1])}")
    return history 

def nelder_mead(f, S, epsilon = 1e-4, alpha = 1, beta = 2, gamma = 0.5):
    S = np.array(S)
    n = S.shape[0]
    history = []
    function_calls = 0
    iter = 0

    def evaluate_simplex(S):
        return np.array([f(S[0, i], S[1, i]) for i in range(S.shape[1])])


    y = evaluate_simplex(S)
    function_calls += n
    delta = np.inf

    while delta > epsilon and iter < ITERATIONS and function_calls < MAX_CALLS:
        iter += 1
        indices = np.argsort(y)
        S = S[:, indices]
        y = y[indices]

        x_l = S[:, 0]
        x_h = S[:, -1]
        x_s = S[:, -2]

        y_l = y[0]
        y_h = y[-1]
        y_s = y[-2]

        x_m = np.mean(S[:, :-1], axis=1)

        x_r = x_m + alpha * (x_m - x_h)
        y_r = f(x_r[0], x_r[1])

        function_calls += 1

        if y_r < y_l:
            x_e = x_m + beta * (x_r - x_m)
            y_e = f(x_e[0], x_e[1])
            function_calls += 1
            if y_e < y_r:
                S[:, -1] = x_e
                y[-1] = y_e
            else:
                S[:, -1] = x_r
                y[-1] = y_r
        elif y_l <= y_r < y_s:
            S[:, -1] = x_r
            y[-1] = y_r
        else:
            if y_r < y_h:
                x_c = x_m + gamma * (x_r - x_m)
            else:
                x_c = x_m + gamma * (x_h - x_m)
            
            y_c = f(x_c[0], x_c[1])
            function_calls += 1

            if y_c < min(y_h, y_r):
                S[:, -1] = x_c
                y[-1] = y_c
            else:
                for i in range(1, S.shape[1]):
                    S[:, i] = x_l + 0.5 * (S[:, i] - x_l)
                    y[i] = f(S[:, i])
                    function_calls += 1
        
        delta = np.std(y)
        history.append((S[0, 0], S[1, 0], y[0]))
    print(f"Minimum nelder mead: {S[:, np.argmin(y)]}, f(x, y) = {f(*S[:, np.argmin(y)])}")
    return history

def quasi_newton_method(f, grad_f, x0, eps=1e-6, max_iter=100, max_calls=1000):
    m = len(x0)
    method = DFP(m)
    method.init(f, grad_f, x0)
    
    x = np.array(x0)
    history = []
    function_calls = 0
    iter = 0

    while function_calls < max_calls and iter < max_iter:
        x_new = method.step(f, grad_f, x)
        history.append((x_new[0], x_new[1], f(x_new[0], x_new[1])))
        x = x_new
        function_calls += 1
        iter += 1
        
        # Podmínka konvergence
        if np.linalg.norm(grad_f(x[0], x[1])) < eps:
            break
    print(f"Minimum quasi newton: x= {x[0]}, y ={x[1]}, f(x,y) = {f(x[0], x[1])}")
    return history

def visualize(f, history, method_name):
        
    X = np.arange(-2, 2, 0.15)
    Y = np.arange(-1, 3, 0.15)
    X, Y = np.meshgrid(X, Y)
    Z = f(X, Y)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, alpha=0.5)

    # Historie gradientního sestupu
    history = np.array(history)

    # Vykreslovani bodu z jednotlivych iteraci
    ax.scatter(START_X, START_Y, f(START_X, START_Y), c='g', marker='o', s=100, label='Starting point')
    ax.scatter(history[:, 0], history[:, 1], history[:, 2], c='k', marker='o', s=10, label='Points')
    ax.scatter(history[-1, 0], history[-1, 1], history[-1, 2], c='r', marker='o', s=100, label='End point')


    # Spojnice mezi body
    ax.plot(np.insert(history[:, 0], 0, START_X), 
            np.insert(history[:, 1], 0, START_Y), 
            np.insert(history[:, 2], 0, f(START_X, START_Y)), 
            color='y', linewidth=1, label='Path')


    # Popisky os
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title(f"{method_name} pro Rosenbrockovu funkci")

    # Nastavení rozsahu osy z
    ax.set_zlim(0, 200)

    # Legenda
    ax.legend()
    plt.show()


if __name__ == "__main__":
    history_cyclic = cyclic_coordinate_descent_with_acceleration_step(f, START_X, START_Y, EPS_CYCLIC)
    history_hooke = hooke_jeeves(f, START_X, START_Y, ALPHA_HOOKE_JEEVES, EPS_HOOKE_JEEVES)
    history_nelder = nelder_mead(f, SIMPLEX_S)
    history_quasi = quasi_newton_method(f, df, [START_X, START_Y])
    print(f"Optimalni minimum pro Rosenbrockovu funkci x_opt = {A}, y_opt = {A**2}, f(x_opt, y_opt) = 0")
    visualize(f, history_cyclic, "Cyclic coordinate descent")
    visualize(f, history_hooke, "Hooke-Jeeves")
    visualize(f, history_nelder, "Nelder-Mead")
    visualize(f, history_quasi, "Quasi-Newton")
    
