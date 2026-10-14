import numpy as np
import matplotlib.pyplot as plt

"""INPUTS para saber qué probabilidad tengo de que sea el yerno 
de mi ('suegra')"""
#Me ve futuro
#Qué tan bien congenio con la familia
#Influencia externa
#La hija me quiere = este input es el que tiene doble peso en el entrenamiento


#Convierte cualquier valor entre 0 y 1
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#Elimina valores negativos
def relu(z):
    return np.maximum(0, z)

#Ayuda a calcular el aprendizaje 
def relu_deriv(z):
    return (z > 0 ).astype(float)

#Calcula el error de la red
def bce(y_real, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred))

np.random.seed(42)

#Pesos y bias de la capa oculta además define cuántas neruonas e inputs
w1 =np.random.randn(6,4) * 0.1
b1 = np.zeros(6)
w2 = np.random.randn(1,6) * 0.1
b2 = np.zeros(1)
#Controla que tan rapido aprende
lr = 0.1

#Hace la predicción multiplicando entradas por pesos más bias
def fd(x):
    z1 = np.dot(w1, x) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)
    #Devuelve los datos del aprendizaje "No los datos finales"
    return a2[0], {'x':x, 'z1':z1, 'a1':a1, 'z2':z2, 'a2':a2}

#Calcula los gradientes para aprender, propaga el error hacia atrás
def bk(cache, y_real):
    a2 = cache['a2']
    a1 = cache['a1']
    z1 = cache['z1']
    x = cache['x']
    dz2 = a2 - y_real
    dw2 = np.outer(dz2, a1)
    db2 = dz2
    da1 = np.dot(w2.T, dz2) 
    dz1 = da1 * relu_deriv(z1)
    dw1 = np.outer(dz1, x)
    db1 = dz1
    #Devuelve los gradientes
    return {'dw1':dw1, 'db1':db1, 'dw2':dw2, 'db2':db2}

#Datos de entrenamiento
X = np.array([
    [0.8, 0.7, 0.6, 0.9],
    [0.9, 0.9, 0.3, 0.9],
    [0.2, 0.3, 0.7, 0.1],
    [0.8, 0.4, 0.9, 0.9],
    [0.1, 0.3, 0.7, 0.1],
    [0.1, 0.7, 0.2, 0.1]
])
#Le da más peso al último input
X[:,3] = X[:,3] * 2.0
  
#Los resultados que se esperan del apredizaje  
y = np.array([1, 1, 0, 1, 0, 0])

#Crea una lista que agrega el historial de pérdidas
h_p = []
for iteracion in range(1000):
    #Guarda predicciones
    prediccion = []
    
    #Crea una matriz del mismo tamaño de w1,b1 etc, acumula los gradientes
    grads_acum = {
        'dw1': np.zeros_like(w1),
        'db1': np.zeros_like(b1),
        'dw2': np.zeros_like(w2),
        'db2': np.zeros_like(b2)
    }
    #Recore cada dato
    for i, xi in enumerate(X):
        pred, cache = fd(xi)
        #Guarda predicción
        prediccion.append(pred)
        #Calcula gradientes
        grads = bk(cache, np.array([y[i]]))
        #Suma gradientes
        for k in grads_acum:
            grads_acum[k] += grads[k] / len(X)
            
    #Calcula qué tan bien aprende la red
    poss = bce(y, np.array(prediccion))
    h_p.append(poss)
    
    #Actualiza los pesos y bias para que la red prediga mejor
    w1 -= lr * grads_acum['dw1']
    b1 -= lr * grads_acum['db1']
    w2 -= lr * grads_acum['dw2']
    b2 -= lr * grads_acum['db2']
    
    #Muestra los resultados cada 100 iteraciones
    if iteracion %100 == 0:
        acc = np.mean((np.array(prediccion) > 0.5) == y) *100
        print(f'Iteracion {iteracion:4d} | pérdida: {poss:.4f} | accuracy {acc:.0f}%')
        
#Resultados
print("\n PREDICCIONES FINALES")
nombres = [
    "Persona Entrenamiento 1",
    "Persona Entrenamiento 2",
    "Persona Entrenamiento 3",
    "Persona Entrenamiento 4",
    "Persona Entrenamiento 5",
    "Persona Entrenamiento 6"
]

for i, xi in enumerate(X):
    pred, _ = fd(xi)
    estado = "Lo mas probable es que si sea mi suegra" if pred > 0.5 else "Lo mas probable es que no sea mi suegra"
    print(f'{nombres[i]:25}: {pred * 100:.1f}% - {estado}')
    
#Yo
k = "keniu"
keniu = np.array([0.8, 0.9, 0.6, 0.4])
keniu[3] = keniu[3] * 2.0
pred, _ = fd(keniu)

print("\nPACIENTE NUEVO:")
if pred > 0.5:
    print(f"{pred*100:.1f}% → Lo mas probable es que si sea suegra de {k}")
else:
    print(f"{pred*100:.1f}% → Lo mas probable es que no sea la suegra de {k}")

# Gráfica, qué tan bien aprende la red
plt.figure(figsize=(8, 4))
plt.plot(h_p, linewidth=2)
plt.xlabel('Época')
plt.ylabel('Pérdida (BCE)')
plt.title('Curva de aprendizaje')
plt.grid(True)
plt.show()

#Esta red aún no aprende solo memoriza los patrones
input("Como estas")