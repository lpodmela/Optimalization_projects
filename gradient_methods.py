import numpy as np
import matplotlib.pyplot as plt

# Nastaveni PP
A = 1
B = 5
ITERATIONS = 100
START_X, START_Y = -1, -1
STOP_CRITERION = 1e-2

# Pro volbu alphy u gradient descent metody -optimalni dle line search metody nebo 0.9^i (false)
OPTIMAL_ALPHA = True
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

def gradient_descent(x0, y0, num_iterations, optimal_alpha):
    x = np.array([x0, y0])
    history = []

    for i in range(num_iterations):
        grad = df(x[0], x[1])  # Výpočet gradientu pro akutalni bod
        
        # Norma gradientu
        grad_norm = np.linalg.norm(grad)
        
        # Kontrola konvergence
        if np.all(np.abs(grad)) < STOP_CRITERION:
            break
        
        d = -grad / grad_norm  # Směr sestupu

        # Volitelna alfa dle zadani a) nebo b) 
        if optimal_alpha:
            alpha = line_search(f, x, d) # Optimalni alfa
        else:
            alpha = 0.9**i
        
        # Aktualizace bodu x ve smeru d, vcetne funkcni hodnoty
        x = x + alpha * d 
        f_value = f(x[0], x[1])

        # Ukládání historie pro vykresleni
        history.append((x[0], x[1], f_value))

    print(f"Konecne nalezene minimum pro gradient descent metodu: x= {x[0]}, y ={x[1]}, f(x,y) = {f_value}")

    return  history

def conjugate_gradient(x0, y0, num_iterations):
    x = np.array([x0, y0])
    history = []
    grad = df(x[0], x[1])  # Počáteční gradient
    d = 0 # Směr sestupu
    prev_grad = grad  
    
    for _ in range(num_iterations):
        grad = df(x[0], x[1]) # Výpočet gradientu
        
        if np.all(np.abs(grad)) < STOP_CRITERION:
            break
        
        # Výpočet bety podle Polak-Ribiere
        beta = max(0, np.dot(np.transpose(grad), grad - prev_grad) / np.dot(np.transpose(prev_grad), prev_grad))
        
        # Aktualizace směru sestupu
        d = -grad + beta * d
        
        # Optimalini alfa z prvniho ukolu
        alpha = line_search(f, x, d)
        
        # Aktualizace bodu + funkcni hodnoty
        x = x + alpha * d
        f_value = f(x[0], x[1])

        history.append((x[0], x[1], f_value))
        
        prev_grad = grad

    print(f"Konecne nalezene minimum pro cojugate gradient: x= {x[0]}, y ={x[1]}, f(x,y) = {f_value}")
    
    return  history

def visualization(path, method_name):
    X = np.arange(-2, 2, 0.15)
    Y = np.arange(-1, 3, 0.15)
    X, Y = np.meshgrid(X, Y)
    Z = f(X, Y)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, alpha=0.5)

    # Historie gradientního sestupu
    path = np.array(path)

    # Vykreslovani bodu z jednotlivych iteraci
    ax.scatter(START_X, START_Y, f(START_X, START_Y), c='g', marker='o', s=100, label='Starting point')
    ax.scatter(path[:, 0], path[:, 1], path[:, 2], c='k', marker='o', s=10, label='Points')
    ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], c='r', marker='o', s=100, label='End point')


    # Spojnice mezi body
    ax.plot(np.insert(path[:, 0], 0, START_X), 
            np.insert(path[:, 1], 0, START_Y), 
            np.insert(path[:, 2], 0, f(START_X, START_Y)), 
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
    grad_desc = gradient_descent(START_X, START_Y, ITERATIONS, OPTIMAL_ALPHA)
    grad_cojn = conjugate_gradient(START_X, START_Y, ITERATIONS)
    visualization(grad_desc, "Gradient descent metoda")
    visualization(grad_cojn, "Conjugate gradient metoda")
    print(f"Optimalni minimum pro Rosenbrockovu funkci x_opt = {A}, y_opt = {A**2}, f(x_opt, y_opt) = 0")    
