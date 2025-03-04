import numpy as np
import matplotlib.pyplot as plt

# Nastaveni PP
A = 1
B = 5


ROUND = 10 # Zaokrouhleni (pocet des. mist) finalniho vysledku (zaokrouhleni pouze pro print)

ITERATIONS = 100
MAX_CALLS = 1000

START_X, START_Y = -1, -1

STOP_CRITERION = 1e-2

DELTA_CYCLIC = 1e-4
EPS_CYCLIC = 1e-6

EPS_HOOKE_JEEVES = 1e-6
ALPHA_HOOKE_JEEVES = 1
GAMA_HOOKE_JEEVES = .5

EPS_NELDER_MEAD = 1e-8
DELTA_NELDER_MEAD = 1e-6
ALPHA_NELDER_MEAD = 1
BETA_NELDER_MEAD = 2
GAMA_NELDER_MEAD = .5

EPS_QUASI_NEWTON = 1e-6

SIMPLEX_S = np.array([
    [-1, -0.8, -0.5],
    [-1, -0.8, -1]
])

# Rosenbrockova funkce + gradient
f = lambda x, y: (A - x)**2 + B * (y - x**2)**2

df = lambda x, y: np.array([-2 * (A - x) - 4 * B * x * (y - x**2), 2 * B * (y - x**2)])


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

def cyclic_coordinate_descent_with_acceleration_step(f, x0, y0, eps = EPS_CYCLIC, max_calls=MAX_CALLS, max_iter = ITERATIONS):
    x = np.array([x0, y0])
    n = len(x)
    delta = DELTA_CYCLIC
    history = []
    function_calls = 0
    iter = 0
    
    while abs(delta) > eps and function_calls < max_calls and iter < max_iter:
        x_prev = x.copy()
        
        for i in range(n):
            if function_calls >= max_calls:
                break
           
            # basis(i, n)
            d = np.eye(n)[i]

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
    print(f"Minimum metodou Cyclic coordinate descent: x = {round(x[0], ROUND)}, y = {round(x[1], ROUND)}, f(x,y) = {round(f(x[0], x[1]), ROUND)}, počet iterací = {iter}, funkčních volání = {function_calls}")
    return history

def hooke_jeeves(f, x0, y0, alpha, eps, gamma = GAMA_HOOKE_JEEVES, max_calls = MAX_CALLS, max_iter = ITERATIONS):
    x = np.array([x0, y0])
    y = f(x[0], x[1])
    n = len(x)
    history = []
    function_calls = 1
    iter = 0

    while alpha > eps and function_calls < max_calls and iter < max_iter:
        improved = False
        x_best, y_best = x, y
        for i in range(n):
            for sgn in (-1, 1):
                x_n = x + sgn * alpha * np.eye(n)[i]
                y_n = f(x_n[0], x_n[1])
                function_calls += 1
                if y_n < y_best:
                    x_best, y_best = x_n, y_n
                    improved = True
        x, y = x_best, y_best
        if not improved:
            alpha *= gamma
        history.append((x[0], x[1], f(x[0], x[1])))
        iter += 1
    print(f"Minimum metodou Hooke-Jeeves: x = {round(x[0], ROUND)}, y = {round(x[1], ROUND)}, f(x,y) = {round(f(x[0], x[1]), ROUND)}, počet iterací = {iter}, funkčních volání = {function_calls}")
    return history 

def nelder_mead(f, S, epsilon = EPS_NELDER_MEAD, alpha = ALPHA_NELDER_MEAD, beta = BETA_NELDER_MEAD, gamma = GAMA_NELDER_MEAD):
    n = S.shape[0]
    history = []
    iter = 0
    function_calls = 0

    y = np.array([f(S[0, i], S[1, i]) for i in range(S.shape[1])])
    function_calls += n
    delta = DELTA_NELDER_MEAD

    while delta > epsilon and iter < ITERATIONS and function_calls < MAX_CALLS:
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
        elif y_r > y_s:
            if y_r <= y_h:
                x_h, y_h, S[:, -1], y[-1] = x_r, y_r, x_r, y_r

            x_c = x_m + gamma * (x_h - x_m)    
            y_c = f(x_c[0], x_c[1])
            function_calls += 1

            if y_c > y_h:
                for i in range(1, len(y)):
                    S[:, i] = (S[:, i] + x_l) /2
                    y[i] = f(S[0, i], S[1,i])
                    function_calls += 1
            else:
                S[:,-1] = x_c
                y[-1] = y_c
        else:
            S[:,-1] = x_r
            y[-1] = y_r
        
        delta = np.std(y)
        history.append((S[0, 0], S[1, 0], y[0]))
        iter += 1
    print(f"Minimum metodou Nelder-Mead: x = {round(S[0, 0], ROUND)}, y = {round(S[1,0], ROUND)}, f(x, y) = {round(y[0], ROUND)}, počet iterací = {iter}, funkčních volání = {function_calls}")
    return history

class DFP:
    def __init__(self, x):
        self.m = len(x)
        self.Q = np.eye(self.m)
        

    def step(self, f, grad_f, x):
        g = grad_f(x[0], x[1])
        d = -np.dot(self.Q, g)
        alpha = line_search(f, x, d) 
        x_new = x + alpha * d  # Nový bod

        g_new = grad_f(x_new[0], x_new[1])  # Nový gradient
        delta_x = x_new - x
        delta_g = g_new - g

        # Aktualizace matice Q, roznasobeni matic (...) * (...)
        self.Q = self.Q - np.outer(np.dot(self.Q, delta_g), np.dot(self.Q, delta_g)) / np.dot(delta_g, np.dot(self.Q, delta_g)) \
                    + np.outer(delta_x, delta_x) / np.dot(delta_x, delta_g)
        
        return x_new

def quasi_newton_method(f, grad_f, x0, eps = EPS_QUASI_NEWTON, max_iter = ITERATIONS, max_calls = MAX_CALLS):
    method = DFP(x0)
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
    print(f"Minimum metodou Quasi-Newton: x = {round(x[0], ROUND)}, y = {round(x[1], ROUND)}, f(x,y) = {round(f(x[0], x[1]), ROUND)}, počet iterací = {iter}, funkčních volání = {function_calls}")
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
    history_cyclic = cyclic_coordinate_descent_with_acceleration_step(f, START_X, START_Y)
    history_hooke = hooke_jeeves(f, START_X, START_Y, ALPHA_HOOKE_JEEVES, EPS_HOOKE_JEEVES)
    history_nelder = nelder_mead(f, SIMPLEX_S)
    history_quasi = quasi_newton_method(f, df, [START_X, START_Y])
    print(f"Optimálni minimum pro Rosenbrockovu funkci x_opt = {A}, y_opt = {A**2}, f(x_opt, y_opt) = 0")
    visualize(f, history_cyclic, "Cyclic coordinate descent")
    visualize(f, history_hooke, "Hooke-Jeeves")
    visualize(f, history_nelder, "Nelder-Mead")
    visualize(f, history_quasi, "Quasi-Newton")
    
