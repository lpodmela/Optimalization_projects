import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (0.2 * np.exp(x - 2) - x)

t = np.linspace(-1, 5, 100000)

def quadratic_fit_search(f, a, b, c, n):
    ya, yb, yc = f(a), f(b), f(c)

    for i in range(1, n - 2):
        x = 0.5 * (ya * (b**2 - c**2) + yb * (c**2 - a**2) + yc * (a**2 - b**2)) / (ya * (b - c) + yb * (c - a) + yc * (a - b))

        yx = f(x)

        if x > b:
            if yx > yb:
                c, yc = x, yx

            else:
                a, ya, b, yb = b, yb, x, yx

        else:
            if yx > yb:
                a, ya = x, yx

            else:
                c, yc, b, yb = b, yb, x, yx
    return a, b, c

a, b, c = quadratic_fit_search(f, -1, 4, 5, 5)

delka = abs(c - a)
print(f"Délka intervalu <a,c>: {delka}")
print(f"Bod a: {a:,.4f}")
print(f"Bod b: {b:,.4f}")
print(f"Bod c: {c:,.4f}")


plt.figure()

plt.plot(t, f(t), 'b', label='f(x)')
plt.scatter(a, f(a), color='red', label='Body a, c')
plt.scatter(c, f(c), color='red')  
plt.scatter(b, f(b), color='blue', label='Bod b')  

# Popisky bodů
plt.text(a, f(a), ' a', fontsize=8, verticalalignment='bottom', horizontalalignment='left', color='black', fontweight='bold')
plt.text(c, f(c), ' c', fontsize=8, verticalalignment='bottom', horizontalalignment='right', color='black', fontweight='bold')  
plt.text(b, f(b), ' b', fontsize=8, verticalalignment='bottom', horizontalalignment='right', color='black', fontweight='bold') 

plt.axhline(0, color='black', linewidth=1)  # Osa x
plt.axvline(0, color='black', linewidth=1)  # Osa y

offset = 0.2 * (max(f(t)) - min(f(t)))  # 20 % výškového rozsahu grafu
plt.vlines(a, ymin=f(a) - offset, ymax=f(a) + offset, color='red', linestyle='--', label='Interval [a, c]')  
plt.vlines(c, ymin=f(c) - offset, ymax=f(c) + offset, color='red', linestyle='--') 

# Přidání kóty mezi a a c
kot_y = max(f(a), f(c)) + 0.3 * offset  
plt.annotate(
    "", xy=(a, kot_y), xytext=(c, kot_y), 
    arrowprops=dict(arrowstyle="<->", color="black")
)
plt.text((a + c) / 2, kot_y + 0.05 * offset, f'{c - a:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xlabel('x')  # Popis osy x
plt.ylabel('f(x)')  # Popis osy y
plt.title(r'Quadratic fit search pro graf funkce $f(x) = 0.2 e^{x - 2} - x$')  # Nadpis grafu
plt.grid(True)  # Mřížka
plt.legend()  # Legenda

plt.show()
