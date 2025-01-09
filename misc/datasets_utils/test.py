import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'



def custom_sigmoid(i, a, b):
    """
    Funkcja oblicza wartość zmodyfikowanej funkcji sigmoidalnej.

    Parametry:
    i : liczba lub numpy array
        Wartość wejściowa lub tablica wartości.
    a : float
        Parametr skalujący nachylenie funkcji.
    b : float
        Parametr przesuwający wykres w osi poziomej.

    Zwraca:
    Wartość funkcji obliczoną dla każdego elementu i.
    """
    return 1 - 1 / (1 + np.exp((i - b) / a))


# Przedział wartości i
i_values = np.linspace(-40, 120, 1000)

# Przykładowe wartości parametrów a i b
params = [(5, 30), (15, 30), (15, 50)]

# Rysowanie wykresów dla różnych parametrów
plt.figure(figsize=(8, 4))
for a, b in params:
    y_values = custom_sigmoid(i_values, a, b)
    plt.plot(i_values, y_values, label=r'$a=' + str(a) + '\quad b=' + str(b) + '$')

# plt.title(r'Wykres funkcji sigmoidalnej dla różnych parametrów $a$ i $b$')
plt.xlabel(r'x')
plt.ylabel(r'Wartość funkcji')
plt.legend()
plt.grid(True)
plt.show()
