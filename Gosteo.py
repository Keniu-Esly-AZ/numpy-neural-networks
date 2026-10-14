import numpy as np
import matplotlib.pyplot as plt

"""Red Neuronal para saber que probabilidad tengo de que me dejen de esponder los mensajes"""
'''Esta red es de 6 inputus y 2 capas ocultas'''

# Tiempo enque me responde
# Ella inicia las conversaciones
# Usa stikers, emogis o carias
# Responde Seco
# Acepta las salidas 
# Responde con entuciasmo, o solo por  responder

#Funciones de activacion
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0,z)

#Funcines para que la red "Mejore"
def relu_deriv(z):
    return (z > 0).astype(float)

def bce(y_real, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred))

np.random.seed(42)

#Pesos y bias de la red ademas de la arquitectura
w1 = np.random.randn(8,6) * np.sqrt(2/6)
b1 = np.zeros(8)
w2 = np.random.randn(6,8) * np.sqrt(2/8)
b2 = np.zeros(6)
w3 = np.random.randn(1,6) * np.sqrt(2/6)
b3 = np.zeros(1)

#Hace las predicciones
def fd(x):
    z1 = np.dot(w1, x) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(w3, a2) + b3
    a3 = sigmoid(z3)
    return a3[0], {'x':x, 'z1':z1, 'a1':a1, 'z2':z2, 'a2':a2, 'z3':z3, 'a3':a3}

#Calcula el aprendizaje
def bk(cache, y_real):
    a3 = cache['a3']
    a2 = cache['a2']
    a1 = cache['a1']
    z2 = cache['z2']
    z1 = cache['z1']
    x = cache['x']
    
    dz3 = a3 - y_real
    dw3 = np.outer(dz3, a2)
    db3 = dz3
    dz2 = np.dot(w3.T, dz3) * relu_deriv(z2)
    dw2 = np.outer(dz2, a1)
    db2 = dz2
    dz1 = np.dot(w2.T, dz2) * relu_deriv(z1)
    dw1 = np.outer(dz1, x)
    db1 = dz1
    return {'dw1':dw1, 'db1':db1, 'dw2':dw2, 'db2':db2, 'dw3':dw3, 'db3':db3}

"""Crea datos falsos para entrenar la red 
ademas de esto esta entrenada con 6000 datos 
falsos para que no de problemascon la arquitectura de la red"""
def Datos_cinteticos(n=5000):
    X = []
    Y = []
    
    #Crea los datos falsos automaticamente
    for _ in range(n):
        tiempo = np.random.rand()
        inicia = np.random.rand()
        emojis = np.random.rand()
        seco = np.random.rand()
        salidas = np.random.rand()
        calidad = np.random.rand()
        
        x = np.array([tiempo, inicia, emojis,
                      seco, salidas, calidad
                      ])
        
        #Decide que tan probable es el ghosteo
        score = (
            inicia * 2 +
            emojis * 1.5 +
            salidas * 2.3 +
            calidad * 3 -
            seco * 3 - 
            tiempo * 2 
        )
        
        #Convierte el score en porcentaje
        y = sigmoid(score - 2)
        
        X.append(x)
        Y.append(y)
        
    return np.array(X), np.array(Y)

#Crea datos de entrenamiento
X,Y = Datos_cinteticos(6000)

#Divide los datos entre los que son para el entrenamiento y lo que son para la prueba
split = int(len(X) * 0.8)
X_train, Y_train = X[:split], Y[:split]
X_val, Y_val = X[split:], Y[split:]

#Velocidad de aprendizaje
lr = 0.001

#Cantidad de vueltas
epochs = 110

#Guarda historial de perdida
h_l = []
v_l = []

#Entrenamiento
for epoch in range(epochs):
    prediccion = []
    
    #Recorre todos los datos
    for x,y in zip(X_train, Y_train):
        pred, cache = fd(x)
        grads = bk(cache, y)
        
        #Actualiza pesos
        w1 -=lr * grads['dw1']
        b1 -=lr * grads['db1']
        w2 -=lr * grads['dw2']
        b2 -=lr * grads['db2']
        w3 -=lr * grads['dw3']
        b3 -=lr * grads['db3']
        
        prediccion.append(pred)
        
    #Calcula perdida
    loss = bce(Y_train, np.array(prediccion))
    h_l.append(loss)
    
    vp = []
    
    #Comruba si aprendio o solo memoriso
    for x in X_val:
        pred, _ = fd(x)
        vp.append(pred)
        
    #Calcula perdida de validacion
    vl = bce(Y_val, np.array(vp))  
    v_l.append(vl)    
    
    #Convierte probabilidades a 0 y 1
    p_b = (np.array(vp) > 0.5).astype(int)
    y_b = (Y_val > 0.5).astype(int)
    
    #Calcula precision
    acc = np.mean(p_b == y_b)
    
    #Muestra resultados
    if epoch % 10 == 0:
        print(f'Iteracion: {epoch:3d} | perdida: {loss:.4f} | validacion: {vl:.4f} | precisión: {acc *100:.1f}%')

#Prueba final
print("\n Probar la red")

mis_datos = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
prediccion, _ = fd(mis_datos)

print(f"\nProbabilidad de ghosteo: {prediccion * 100:.2f}%")

#Resultado final
if prediccion >= 0.5:
    print("❌ Alta probabilidad de ghosteo")
else:
    print("✔ Baja probabilidad de ghosteo")

#Grafica del aprendizaje
plt.plot(h_l, label="train_loss")
plt.plot(v_l, label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()