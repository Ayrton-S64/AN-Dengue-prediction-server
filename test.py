import matplotlib.pyplot as plt

# Datos
variables = [
    "id", "vómitos", "escalofríos", "sangrado de encías", "problemas de habla", "rigidez de cuello",
    "fatiga", "restricción respiratoria", "erupción en forma de diana", "irritación en los labios",
    "inflamación de los dedos", "inflamación", "sangrado gastrointestinal", "ojos amarillos",
    "postración", "diarrea", "inflamación del dedo del pie", "escalofríos", "pérdida de peso",
    "hipoglucemia", "dolor orbital", "hinchazón de los ganglios", "sangrado en la boca", "convulsiones",
    "microcefalia", "ritmo cardíaco lento", "dolor de espalda", "sabor amargo en la lengua", "ojos rojos",
    "dolor muscular", "confusión", "temblores", "hinchazón", "parálisis", "dolor de cuello", "mialgia",
    "anemia", "problemas de digestión", "efusión pleural", "irritabilidad",
    "orina parecida a la de cola de cola", "hiperpirexia", "distorsión facial", "mareos", "debilidad",
    "dolor abdominal", "erupción", "piel amarilla", "lesiones en la piel", "sensibilidad a la luz",
    "pérdida de apetito", "dolor de cabeza", "ictericia", "coma", "pérdida de la micción",
    "dolor de estómago", "dolor en las articulaciones", "úlceras", "fiebre repentina", "hipotensión",
    "pérdida de uñas de los dedos del pie", "náuseas", "picazón", "ascitis", "sangrado nasal"
]

coeficientes = [
    -0.0000764, -0.00253, 0.0340, -0.0548, -0.0703, -0.0788, -0.0796, -0.0898, -0.0928, -0.0973, -0.1078,
    -0.1129, -0.1421, -0.1506, -0.1560, 0.1569, -0.1594, -0.1609, -0.1685, -0.1692, -0.1753, -0.1768, 0.1812,
    -0.1813, -0.1853, -0.1916, -0.1922, -0.1944, -0.2150, 0.2279, -0.2297, -0.2352, -0.2405, -0.2434, -0.2497,
    -0.2545, -0.2564, -0.2609, -0.2615, -0.2633, -0.2651, -0.2711, -0.2814, -0.2821, -0.2880, -0.2940, 0.3200,
    -0.3234, -0.3287, -0.3402, -0.3585, 0.3615, -0.3812, -0.3817, -0.4164, -0.4493, 0.4660, -0.4828, 0.4899,
    -0.4926, -0.5073, 0.5366, -0.5413, -0.5777, 0.6327
]

# Crear gráfica
plt.figure(figsize=(12, 6))
plt.barh(variables, coeficientes)  # Usamos barh para barras horizontales
plt.xlabel('Coeficiente')
plt.ylabel('Variable')
plt.title('Importancia de los coeficientes de las variables en el modelo de regresión lineal')
plt.tight_layout()

# Mostrar gráfica
plt.show()
