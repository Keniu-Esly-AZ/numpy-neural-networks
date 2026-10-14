import numpy as np
import matplotlib.pyplot as plt

"""Red que predice que lenguajue de programacion mas te conviene aprender"""
'''Todas las redes neuronales que hemos visto hasta ahora solo pueden dar una
respuesta de si o no, con softmax podemos agregar mas opciones haciendo que
tenga un porcentaje del total.'''


"Entradas = Que te atrae?"
#IA
#Diseño y desarrollo web
#Creacion de videojuegos
#Desarrollo de apliaciones
#Bases de Datos
#Redimiento de software

np.random.seed(42)

#Funciones de activacion
def relu(z):
    return np.maximum(0,z)

def relu_deriv(z):
    return(z > 0).astype(float)

#Esta el la funcion que convierte los resultados en probabilidades
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

#Calcula que tanto se equivoco la red
def cross_entropy(y_real, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
    return -np.mean(np.sum(y_real * np.log(y_pred), axis=1))

#Datos de entrenamiento 
X = np.array([
    [10,2,1,2,10,2],[9,1,2,1,9,2],[8,3,2,2,8,1],
    [2,10,3,5,2,3],[1,9,2,4,2,2],[3,8,1,6,1,2],
    [2,2,10,3,1,9],[1,3,9,2,1,10],[2,1,8,3,2,8],
    [3,5,2,10,3,4],[2,4,1,9,2,3],[3,6,2,8,1,2],
    [4,3,9,8,2,5],[3,2,8,9,1,4],[2,3,10,8,2,4],
    [6,2,3,2,5,10],[5,1,2,1,4,9],[7,2,3,2,5,9]
], dtype=float)

#Normalización de los datos
X = X / 10

"Opciones de lenjuagues"
#Python
#Javascript
#C++
#Java-Kotlin
#C#
#SQL

#Resultados que se esperan del aprendizaje 
y = np.array([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5])

#Convierte cada número en un vector lleno de ceros con solo 1 en la posicion de ese numero
Y = np.eye(6)[y]

#Red neuronal de 6,12,8,6
W1 = np.random.randn(6,12) * 0.1
b1 = np.zeros((1, 12))
W2 = np.random.randn(12,8) * 0.1
b2 = np.zeros((1,8))
W3 = np.random.randn(8,6) * 0.1
b3 = np.zeros((1,6))

#Propacion hacia adelante
def fd(x,w1,b1,w2,b2,w3,b3):
    z1 = x @ w1 +b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    a2 = relu(z2)
    z3 = a2 @ w3 + b3
    a3 = softmax(z3)
    
    return z1,a1,z2,a2,z3,a3

#Propagacion hacia atras
def bk(X, Y, z1, a1, z2, a2, a3, W2, W3):
    dz3 = a3 - Y
    dW3 = a2.T @ dz3 / len(X)
    db3 = np.mean(dz3, axis=0, keepdims=True)
    da2 = dz3 @ W3.T
    dz2 = da2 * relu_deriv(z2)
    dW2 = a1.T @ dz2 / len(X)
    db2 = np.mean(dz2, axis=0, keepdims=True)
    da1 = dz2 @ W2.T
    dz1 = da1 * relu_deriv(z1)
    dW1 = X.T @ dz1 / len(X)
    db1 = np.mean(dz1, axis=0, keepdims=True)
    return dW1, db1, dW2, db2, dW3, db3

#Entrenamiento de una red neuronal 
def entrenar(X, Y, epochs=5000, lr=0.05):
    #Estas variables pertenecen al programa 
    global W1, b1, W2, b2, W3, b3#Arquitectura redes neuronales

    #Entrenamiento de la red neuronal
    for epoch in range(epochs):
        #Propagacion hacia adelante
        z1, a1, z2, a2, z3, a3 = fd(X, W1, b1, W2, b2, W3, b3)
        #Calcula el error de la red
        loss = cross_entropy(Y, a3)
        #Propagacion hacia atras
        dW1, db1, dW2, db2, dW3, db3 = bk(
            X, Y, z1, a1, z2, a2, a3, W2, W3)

        #Actualiza los pesos y bias
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        W3 -= lr * dW3
        b3 -= lr * db3

        #Muestra el progreso del aprendizaje
        if epoch % 500 == 0:
            pred = np.argmax(a3, axis=1)
            acc = np.mean(pred == y) * 100
            print(f"Epoca {epoch:4d} | "f"Perdida: {loss:.4f} | "f"Precision: {acc:.1f}%")
            

#Ejecuta el entrenamiento
entrenar(X,Y)            

#Predicion de perosona >X)
nuevo_usuario = np.array([[6, 3, 5, 2, 5, 3]]) / 10

#Propagacion hacia adelante
resultado = fd(nuevo_usuario, W1, b1, W2, b2, W3, b3)[-1][0]
lenguajes = ["Python","JavaScript","C++","Java","C#","SQL"]

#Muestra las probabilidades
print("\nProbabilidades:")
for lenguaje, prob in zip(lenguajes, resultado):
    print(f"{lenguaje:12s}: {prob*100:.2f}%")
fig, ax = plt.subplots(figsize=(11, 7))
input_nodes = [(1, i) for i in np.linspace(0.5, 6, 6)]
hidden1_nodes = [(3, i) for i in np.linspace(0.3, 6.2, 12)]
hidden2_nodes = [(5, i) for i in np.linspace(0.5, 6, 8)]
output_nodes = [(7, i) for i in np.linspace(0.5, 6, 6)]

for i_node in input_nodes:
    for h_node in hidden1_nodes:
        ax.plot([i_node[0], h_node[0]], [i_node[1], h_node[1]], "gray", zorder=1)
for h1_node in hidden1_nodes:
    for h2_node in hidden2_nodes:
        ax.plot([h1_node[0], h2_node[0]], [h1_node[1], h2_node[1]], "gray", zorder=1)
for h2_node in hidden2_nodes:
    for o_node in output_nodes:
        ax.plot([h2_node[0], o_node[0]], [h2_node[1], o_node[1]], "gray", zorder=1)
for node in input_nodes:
    ax.scatter(*node, s=800, color="skyblue", zorder=2)
for node in hidden1_nodes:
    ax.scatter(*node, s=500, color="lightgreen", zorder=2)
for node in hidden2_nodes:
    ax.scatter(*node, s=700, color="lightgreen", zorder=2)
for node in output_nodes:
    ax.scatter(*node, s=800, color="salmon", zorder=2)

nombres = [
    "Inteligencia artificial",
    "Desarrollo web",
    "Desarrollo de videojuegos",
    "Aplicaciones móviles",
    "Bases de datos",
    "Rendimiento de software"
]

for i, nombre in enumerate(nombres):
    ax.text(0.8, input_nodes[i][1], nombre,
            ha="right", va="center", fontsize=9)

lenguajes = ["Python", "JavaScript", "C++", "Java", "C#", "SQL"]

for i, lenguaje in enumerate(lenguajes):
    ax.text(7.3, output_nodes[i][1], lenguaje,
            ha="left", va="center", fontsize=9)

ax.text(1, 6.7, "Entrada\n6", ha="center", fontsize=11)
ax.text(3, 6.7, "Oculta 1\n12", ha="center", fontsize=11)
ax.text(5, 6.7, "Oculta 2\n8", ha="center", fontsize=11)
ax.text(7, 6.7, "Salida\n6", ha="center", fontsize=11)
ax.set_xlim(-2.5, 9)
ax.set_ylim(0, 7.3)
ax.axis("off")
plt.title("Red neuronal 6 → 12 → 8 → 6")
plt.show()