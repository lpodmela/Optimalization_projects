import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (0.2 * np.exp(x - 2) - x)

t = np.linspace(-1, 5, 100000)

def golden_section_search(f, a, b, n):
    fi = (1 + np.sqrt(5)) / 2
    ro = fi - 1
    d = ro * b + (1 - ro) * a
    yd = f(d)

    for i in range(1, n):
        c = ro * a + (1 - ro) * b
        yc = f(c)

        if yc < yd:
            b, d, yd = d, c, yc
        else:
            a, b = b, c
    
    if a < b:
        return a, b
    else:
        a, b = b, a
        return a, b
    
a, b = golden_section_search(f, -1, 5, 5)

delka = abs(b - a)
print(f"Délka intervalu <a,b>: {delka:,.4f}")
print(f"Bod a: {a:,.4f}")
print(f"Bod b: {b:,.4f}")






plt.figure()

plt.plot(t, f(t), 'b', label='f(x)')
plt.scatter(a, f(a), color='red', label='Body a, b')
plt.scatter(b, f(b), color='red')

plt.text(a, f(a), ' a', fontsize=8, verticalalignment='bottom', horizontalalignment='left', color='black', fontweight='bold')
plt.text(b, f(b), ' b', fontsize=8, verticalalignment='bottom', horizontalalignment='right', color='black', fontweight='bold')

plt.axhline(0, color='black', linewidth=1)  # Osa x
plt.axvline(0, color='black', linewidth=1)  # Osa y


offset = 0.2 * (max(f(t)) - min(f(t)))  # 20 % výškového rozsahu grafu
plt.vlines(a, ymin=f(a) - offset, ymax=f(a) + offset, color='red', linestyle='--', label='Interval [a, b]')
plt.vlines(b, ymin=f(b) - offset, ymax=f(b) + offset, color='red', linestyle='--')

# Přidání kóty mezi a a b
kot_y = max(f(a), f(b)) + 0.3 * offset  # Posuneme číslo výš
plt.annotate(
    "", xy=(a, kot_y), xytext=(b, kot_y), 
    arrowprops=dict(arrowstyle="<->", color="black")
)
plt.text((a + b) / 2, kot_y + 0.05 * offset, f'{b - a:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xlabel('x')  # Popis osy x
plt.ylabel('f(x)')  # Popis osy y
plt.title(r'Golden section search pro graf funkce $f(x) = 0.2 e^{x - 2} - x$')  # Nadpis grafu
plt.grid(True)  # Mřížka
plt.legend()  # Legenda

plt.show()