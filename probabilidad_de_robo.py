import numpy as np
import matplotlib.pyplot as plt

"""Esta predice si alguiente va a robar"""
'''Hasta hora mediamos que tan bien aprendia la red con loss y con accuaty
pero la red puede memorizar los datos; esomo si estudiaramos con las
preguntas que van a venir en el examen
'''

"Entradas"
#Que hora es
#Donde estas = lugar
#Tienes cosas valiosas a la vista
#Hay personas cerca
#Que tono de piel tiene la persona ("humor")

np.random.seed(42)
num_muestras = 1200
X = np.zeros((num_muestras, 5))
Y = np.zeros(num_muestras)

for i in range(num_muestras):
    hora = np.random.randint(0, 24)
    lugar = np.random.randint(1, 4)#1=Seguro, 2=Regular, 3=Callejón
    valiosas = np.random.randint(0, 2)# 0=No, 1=Sí
    personas = np.random.randint(0, 51)     
    tono_piel = np.random.randint(1, 11)
    #Mientas mas de noche mas probabiliades de que te roben
    noche = 1 if hora >= 21 or hora <= 5 else 0
    score = (
        0.55 * noche
        + 0.85 * (lugar - 1)
        + 1.45 * valiosas
        - 0.04 * personas
        + 0.35 * tono_piel
        - 2.80
        + np.random.randn() * 0.40
    )

    prob = 1 / (1 + np.exp(-score))
    #Decidde si al final roban 1 si, 0 no.
    Y[i] = int(np.random.rand() < prob)
    #Guarda las entradas en una lista de datos 
    X[i] = [hora, lugar, valiosas, personas, tono_piel]

'''Se utilizan del 100% de los datos 60% entrar, 
20% para validar y 20% para el examen final'''
#Mesclamos dos datos apra que la red no aprenda el orden especifico
idx = np.random.permutation(len(X))
#Calculamos los cortes para separar los datos
c1 = int(len(X) * 0.6)
c2 = int(len(X) * 0.8)

X_tr, Y_tr = X[idx[:c1]], Y[idx[:c1]]#60%
X_va, Y_va = X[idx[c1:c2]], Y[idx[c1:c2]]#20%
X_te, Y_te = X[idx[c2:]], Y[idx[c2:]]#20%

#Funciones de activacion
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

#Propagacion hacia adelante
def fd(x, W1, b1, W2, b2):
    z1 = W1 @ x + b1
    a1 = relu(z1)
    z2 = W2 @ a1 + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2
#Progagacion hacia atras
def bk(x, y, z1, a1, z2, a2, W2):
    dz2 = a2 - y
    dW2 = dz2 * a1
    db2 = dz2
    da1 = dz2 * W2
    dz1 = da1 * (z1 > 0).astype(float)
    dW1 = np.outer(dz1, x)
    db1 = dz1
    return dW1, db1, dW2, db2

#Evita que el codijo explote si intentamos dividir
def dividir_seguro(a, b):
    return 0.0 if b == 0 else a / b

# Evaluación integral que extrae Loss y F1 en una sola pasada
#Revisa que tan bien va la red calculando su error y sus reales
def evaluar_rendimiento(X_set, Y_set, W1, b1, W2, b2, umbral=0.5):
    perdidas = []
    tp, fp, fn = 0, 0, 0
    
    for x, y in zip(X_set, Y_set):#Hace una prediccion para cada muestra
        _, _, _, a2 = fd(x, W1, b1, W2, b2)
        
        # Guardar Loss(Que tan lejos se quedo del restado real)
        a2_clip = np.clip(a2, 1e-7, 1 - 1e-7)
        perdidas.append(-(y * np.log(a2_clip) + (1 - y) * np.log(1 - a2_clip)))
        
        # Mapear predicción binaria para la Matriz de Confusión
        pred = 1 if a2 >= umbral else 0#Si pasa del 50%asumimos que nos van a robar
        if y == 1 and pred == 1: tp += 1
        elif y == 0 and pred == 1: fp += 1
        elif y == 1 and pred == 0: fn += 1
            
    #Calcula las metricas finales usando los datos acumulados
    #Evalua si predijo bien o mal, la comparacion mas importante se hace mas adelante
    loss_promedio = np.mean(perdidas)
    precision = dividir_seguro(tp, tp + fp)
    recall = dividir_seguro(tp, tp + fn)
    f1 = dividir_seguro(2 * precision * recall, precision + recall)
    
    return loss_promedio, f1

#ENTRENAMIENTO Y CORRECION DE LA RED
#Iniciallizamos los pesos y bias con valores aleatorios 
W1 = np.random.randn(6, 5) * np.sqrt(2/5)
b1 = np.zeros(6)
W2 = np.random.randn(6) * np.sqrt(2/6)
b2 = 0.0

alpha = 0.005 #Velocidad a la que aprende la red
epocas = 150 #epoca 

"""Listas vacías para ir guardando como aprende y evoluaciona la red
en cada epoca"""
hist_train_loss, hist_val_loss = [], []
hist_train_f1, hist_val_f1 = [], []

#El bucle principal de aprendizaje("ENTRENAMIENTO")
for epoch in range(epocas):
    #Mesclamos los datos en cada epoca
    orden = np.random.permutation(len(X_tr))
    for i in orden:
        #Calcula el error y corrige lso pesos
        z1, a1, z2, a2 = fd(X_tr[i], W1, b1, W2, b2)
        dW1, db1, dW2, db2 = bk(X_tr[i], Y_tr[i], z1, a1, z2, a2, W2)
        W1 -= alpha * dW1
        b1 -= alpha * db1
        W2 -= alpha * dW2
        b2 -= alpha * db2

    #Ver el estado real en cada época
    loss_tr, f1_tr = evaluar_rendimiento(X_tr, Y_tr, W1, b1, W2, b2)
    loss_va, f1_va = evaluar_rendimiento(X_va, Y_va, W1, b1, W2, b2)
    
    #Guardamos los resultados en el historial
    hist_train_loss.append(loss_tr)
    hist_val_loss.append(loss_va)
    hist_train_f1.append(f1_tr)
    hist_val_f1.append(f1_va)

    if (epoch + 1) % 15 == 0:
        print(f"Época {epoch + 1:3d} | Train F1: {f1_tr:.3f} | Val F1: {f1_va:.3f}")

print("\n¡Entrenamiento completado!")

#PERSONA EJEMPLO
hora = 23; lugar = 3
cosas_valiosas = 3; personas_cerca = 2
piel_noto = 8

persona_ejemplo = np.array([hora, lugar, cosas_valiosas, personas_cerca, piel_noto])
_, _, _, resultado_red = fd(persona_ejemplo, W1, b1, W2, b2)
probabilidad_robo = resultado_red * 100
print(f"\nProbabilidad?????")
print(f"Probabilidad que tengo de sufrir yo un robo: {probabilidad_robo:.2f}%")

if resultado_red >= 0.5:
    print("Lo mas probable es que te chambeen ")
else:
    print("Posiblemente la libraste esta vez")

#GRAFICA
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
#El Error (Loss) 
ax1.plot(range(1, epocas + 1), hist_train_loss, label="Train Loss", color="royalblue", linewidth=2.5)
ax1.plot(range(1, epocas + 1), hist_val_loss, label="Val Loss", color="crimson", linewidth=2.5, linestyle="--")
ax1.set_title("Curva de Pérdida (BCE Loss)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Épocas")
ax1.set_ylabel("Error")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend()
#Rendimiento Real (F1-Score) 
ax2.plot(range(1, epocas + 1), hist_train_f1, label="Train F1-Score", color="forestgreen", linewidth=2.5)
ax2.plot(range(1, epocas + 1), hist_val_f1, label="Val F1-Score", color="darkorange", linewidth=2.5, linestyle="--")
ax2.set_title("Evolución de la Métrica Real (F1-Score)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Épocas")
ax2.set_ylabel("F1 Score (0 a 1)")
ax2.set_ylim(0, 1)  # La escala de éxito siempre va de 0 a 1
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="lower right")
plt.suptitle("Evaluación de Aprendizaje de Verdad (Historial de 4 Métricas)", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


#Grafica 2
fig, ax = plt.subplots(figsize=(10, 7))

input_nodes = [(1, 1.5), (1, 2.3), (1, 3.1), (1, 3.9), (1, 4.7)]
hidden_nodes = [(4, 1.1), (4, 1.9), (4, 2.7), (4, 3.5), (4, 4.3), (4, 5.1)]
output_node = (7, 3.1)

for i_node in input_nodes:
    for h_node in hidden_nodes:
        ax.plot([i_node[0], h_node[0]], [i_node[1], h_node[1]], "gray", alpha=0.4, zorder=1)
for h_node in hidden_nodes:
    ax.plot([h_node[0], output_node[0]], [h_node[1], output_node[1]], "gray", alpha=0.4, zorder=1)
for node in input_nodes:
    ax.scatter(*node, s=850, color="skyblue", edgecolors="black", linewidths=1.5, zorder=2)
for node in hidden_nodes:
    ax.scatter(*node, s=850, color="lightgreen", edgecolors="black", linewidths=1.5, zorder=2)
ax.scatter(*output_node, s=1100, color="salmon", edgecolors="black", linewidths=1.5, zorder=2)

nombres = [
    "¿Qué tono de piel tiene?",
    "¿Hay personas cerca?",
    "¿Cosas valiosas a la vista?",
    "¿Dónde estás? (Lugar)",
    "¿Qué hora es?"
]

for i, nombre in enumerate(nombres):
    ax.text(0.8, input_nodes[i][1], nombre, ha="right", va="center", fontsize=10, fontweight="bold")

ax.text(1, 5.5, "Entrada\n5", ha="center", fontsize=12, fontweight="bold")
ax.text(4, 5.7, "Oculta (ReLU)\n6", ha="center", fontsize=12, fontweight="bold")
ax.text(7, 3.9, "Salida (Sigmoide)\n1", ha="center", fontsize=12, fontweight="bold")
ax.set_xlim(-1, 8.5)
ax.set_ylim(0.5, 6.2)
ax.axis("off")
plt.title("Arquitectura del Evaluador de Riesgo de Robo (5 → 6 → 1)", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.show()
