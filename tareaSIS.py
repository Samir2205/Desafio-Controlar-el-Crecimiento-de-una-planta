import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, CubicSpline
from scipy.optimize import curve_fit

# Datos de días y alturas (en cm)
dias = np.array([7, 21, 35, 49, 63, 77, 91])
alturas = np.array([5.5, 16, 41, 65, 78, 84, 85])

# === Interpolación de Lagrange ===
lag_poly = lagrange(dias, alturas)

# === Interpolación Spline cúbica ===
spline = CubicSpline(dias, alturas)

# === Modelo de regresión logística ===
def modelo_logistico(t, a, b):
    H = 88  # altura máxima asumida
    return H / (1 + np.exp(-(a + b * t)))

params, _ = curve_fit(modelo_logistico, dias, alturas)
a_est, b_est = params

# === Interpolación de Newton ===
def diferencias_divididas(x, y):
    n = len(x)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j - 1]) / (x[j:n] - x[j - 1])
    return coef

def polinomio_newton(x_data, coef, x):
    n = len(coef) - 1
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n - k] + (x - x_data[n - k]) * p
    return p

coef_newton = diferencias_divididas(dias, alturas)

t_eval = np.linspace(7, 91, 500)
y_lagrange = lag_poly(t_eval)
y_spline = spline(t_eval)
y_logistic = modelo_logistico(t_eval, a_est, b_est)
y_newton = polinomio_newton(dias, coef_newton, t_eval)

prediccion_dia_40 = modelo_logistico(40, a_est, b_est)

print("📈 Crecimiento de la planta entre los dias:")
for i in range(1, len(dias)):
    incremento = alturas[i] - alturas[i - 1]
    print(f"🌱 Día {dias[i-1]} ➝ Día {dias[i]}: +{incremento:.2f} cm")

plt.figure(figsize=(10, 6))
plt.plot(dias, alturas, 'o', label='Datos reales', markersize=8, color='green')
plt.plot(t_eval, y_lagrange, '--', label='Lagrange')
plt.plot(t_eval, y_spline, '-.', label='Spline cúbico')
plt.plot(t_eval, y_logistic, '-', label='Regresión logística', color='purple')
plt.plot(t_eval, y_newton, ':', label='Newton', color='orange')
plt.axvline(40, color='gray', linestyle=':', label=f'Día 40: {prediccion_dia_40:.2f} cm')
plt.xlabel("Días")
plt.ylabel("Altura (cm)")
plt.title("Comparación de métodos de ajuste de crecimiento vegetal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
