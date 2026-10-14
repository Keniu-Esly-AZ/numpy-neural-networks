import numpy as np
import matplotlib.pyplot as plt
#Inputs que decidiran que tanta probilidad tengo de ser despedido
#Que tanto rindo en la chamba
#Que tan buena es mi relación con el jefe
#Soy responsable o no
#Que tan buena es mi relación con mis compañeros
#Doy resultados en el trabajo
#Borre la base de datos de la empresa( Si lo hice )
#Funciones de activacion
def sigmoid(z): return 1 / (1 + np.exp(-z))
def relu(z): return np.maximum(0, z)
#Funciones para que la red mejore
def relu_deriv(z): return (z > 0).astype(float)
def bce(y_real, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred))
#Hace la predicción
def forward(x, w1, b1, w2, b2):
    z1 = np.dot(w1, x) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)
    return a2[0], {'x': x, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
#Calcula los gradientes para aprender, calcula matematicamente como debe aprender la red
def backward(cache, y_real, w2):
    a2 = cache['a2']
    a1 = cache['a1']
    z1 = cache['z1']
    x = cache['x']
    dz2 = a2 - y_real
    dw2 = np.outer(dz2, a1)
    db2 = dz2
    #Propaga el error para atras
    da1 = np.dot(w2.T, dz2)
    dz1 = da1 * relu_deriv(z1)
    dw1 = np.outer(dz1, x)
    db1 = dz1
    return {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}
#Son datos sinteticos porque quiero que la red aprenda y no solo memorize los resultados ademas de agregarle ruido apra que no todos los datos sean los mismos y la red pueda apreder
np.random.seed(42)
#Creo artificialmente 4000 datos falsos
datos = 4000
#Genera datos aleatorios para entrenar la red
rendimiento = np.random.uniform(0,1,datos)
relaccion_jefe = np.random.uniform(0,1,datos)
responsabilidad = np.random.uniform(0,1,datos)
relaccion_compañeros = np.random.uniform(0,1,datos)
resultados_trabajo = np.random.uniform(0,1,datos)
borrar_base_datos = np.random.uniform(0,1,datos)
#Le da mas peso al ultimo input
borrar_base_datos *= 3.0
#Une todos los inputs en una sola matriz
X_total = np.column_stack([
    rendimiento,
    relaccion_jefe,
    responsabilidad,
    relaccion_compañeros,
    resultados_trabajo,
    borrar_base_datos
])
#Calcula el porcentaje de ser despedido
puntaje = (
    (1 - rendimiento) +
    (1 - relaccion_jefe) +
    (1 - responsabilidad) +
    (1 - relaccion_compañeros) +
    (1 - resultados_trabajo) +
    borrar_base_datos
)
#Agreha rudio aleatorio
puntaje += np.random.normal(0,0.35,datos)
#Convierte el puntaje en probabilidad osea %
prob_despido = sigmoid(puntaje - 3.5)
#Genera los resultados esperados
y_total = (np.random.rand(datos) < prob_despido).astype(float)
#Muestra cuantos fueron despedidos
print(f"Despidos: {int(y_total.sum())}")
print(f"Se quedan: {int((y_total == 0).sum())}")
#Divide los datos del entrenamiento y de validacion
corte = int(datos * 0.8)
X_train, y_train = X_total[:corte], y_total[:corte]
X_val, y_val = X_total[corte:], y_total[corte:]
#Arquitectura de la red lo que cambia es la multiplicacion
np.random.seed(42)
entradas, capa1 = 6, 5
w1 = np.random.randn(capa1, entradas) * np.sqrt(2/entradas)
b1 = np.zeros(capa1)
w2 = np.random.randn(1, capa1) * np.sqrt(2/capa1)
b2 = np.zeros(1)
lr = 0.02
#Guarda el historial de la perdida
hist_train = []
hist_val = []
#Esta parte ya la explique
for epoca in range(3000):
    preds_train = []
    #Crea una matriz del mismo tamaño de w1,b1 etc, acumula los gradientes
    G = {
        'dw1': np.zeros_like(w1),
        'db1': np.zeros_like(b1),
        'dw2': np.zeros_like(w2),
        'db2': np.zeros_like(b2)
    }
    #Recorre cada cato, aquii comienza apreder pero aun no se corrige
    for i, xi in enumerate(X_train):
        p, cache = forward(xi, w1, b1, w2, b2)
        preds_train.append(p)
        g = backward(cache, np.array([y_train[i]]), w2)
        for k in G:
            G[k] += g[k] / len(X_train)
    #Aqui ya aprende y se corrige sola
    w1 -= lr * G['dw1']
    b1 -= lr * G['db1']
    w2 -= lr * G['dw2']
    b2 -= lr * G['db2']
    hist_train.append(bce(y_train, np.array(preds_train)))
    #Comprueba si realmente aprendio o solo memorizo
    preds_val = []
    for xi in X_val:
        p, _ = forward(xi, w1, b1, w2, b2)
        preds_val.append(p)
    hist_val.append(bce(y_val, np.array(preds_val)))
    #Muestra resultados cada 300 epocas
    if epoca % 300 == 0:
        acc = np.mean((np.array(preds_train) > 0.5) == y_train) * 100
        print(f"Epoca {epoca:4d} | train: {hist_train[-1]:.4f} | val: {hist_val[-1]:.4f} | acc: {acc:.0f}%")
#Grafica donde la red aprende y tambien memoriza los resultados
plt.figure(figsize=(10, 5))
plt.plot(hist_train, label='Train loss', color='#4d9fea', linewidth=2, linestyle='--')
plt.xlabel('Epoca')
plt.ylabel('Pérdida')
plt.title('Preentrenamiento antes de los datos reales.')
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
#Mis datos para corroborar que la red aprenda bien
yo = np.array([0.6, 0.5, 0.1, 0.0, 0.7, 0.1])
yo[5] = yo[5] * 3.0
pred, _ = forward(yo, w1, b1, w2, b2)
print("Iniciando ........")
print(f"Probabilidad de que me despidan: {pred * 100:.2f}%")
