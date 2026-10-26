import sqlite3
import numpy as np
import matplotlib.pyplot as plt

"""Como saber que una persona es un NPC"""
''''El Z-Score es una técnica que unifica la escala de todos tus datos al convertir el
promedio en 0 y medir los demás valores según qué tan lejos o cerca están de ese centro'''

"""El Z-Score es como comparar un elefante y una hormiga; en lugar de compararlos por
su peso en kilos, los comparas diciendo qué tan grandes son respecto al promedio de
su propia especie."""

"Entradas"
#Falta de opinión propia
#Mismas frases de siempre
#Sigue tendencias
#Respuestas automáticas
#Pasatiempos genericos 
#Siempre esta de fondo ("Haciendo bulto")
#No entiende lenjuague avanzado (sarcasmo, ironia)
#Aparece solo cuando lo necesitas (Dador de misiones)

np.random.seed(42)

#Crea un arcchivo de base de datos
conexion = sqlite3.connect("npc.db")
cursor = conexion.cursor()#Permite esciribir en la base de datos

# Borra la tabla si ya existia
cursor.execute("DROP TABLE IF EXISTS personas")

# Creamos la tabla con las 8 señales de un NPC y su categoría final
cursor.execute("""
    CREATE TABLE personas (
        id INTEGER PRIMARY KEY,
        falta_opinion REAL,
        mismas_frases REAL,
        sigue_tendencias REAL,
        respuestas_automaticas REAL,
        pasatiempos_genericos REAL,
        siempre_fondo REAL,
        lenguaje_avanzado REAL,
        aparece_necesitas REAL,
        categoria INTEGER
    )
""")

filas = []

#Inventamos 2000 personas para nuestra base de datos
for i in range(2000):
    #Mientras mas alto el valor mayor es esa cualidad
    falta_opinion = np.random.randint(0, 11)
    mismas_frases = np.random.randint(0, 11)
    sigue_tendencias = np.random.randint(0, 11)
    respuestas_automaticas = np.random.randint(0, 11)
    pasatiempos_genericos = np.random.randint(0, 11)
    siempre_fondo = np.random.randint(0, 11)
    #Mientras mas alto el valor menos comprende el lenjuage avanzado
    lenguaje_avanzado = np.random.randint(0, 11)
    aparece_necesitas = np.random.randint(0, 11)

    score = (
        1.3 * falta_opinion
        + 1.2 * mismas_frases
        + 1.0 * sigue_tendencias
        + 1.2 * respuestas_automaticas
        + 0.7 * pasatiempos_genericos
        + 1.0 * siempre_fondo
        + 1.1 * lenguaje_avanzado
        + 0.8 * aparece_necesitas
    ) / 8.3

    # Clasificamos según el puntaje obtenido
    # 0 = Es un NPC, 1 = Se está convirtiendo, 2 = No es un NPC
    if score >= 6.5:
        categoria = 0
    elif score >= 4.2:
        categoria = 1
    else:
        categoria = 2

    valores = [
        falta_opinion,
        mismas_frases,
        sigue_tendencias,
        respuestas_automaticas,
        pasatiempos_genericos,
        siempre_fondo,
        lenguaje_avanzado,
        aparece_necesitas
    ]

    # Simulamos un 5% de datos faltantes (Cosas vacias que pasan en la vida real)
    for j in range(len(valores)):
        if np.random.rand() < 0.05:
            valores[j] = None

    filas.append((i, *valores, categoria))

# Guardamos todos los datos en la DB
cursor.executemany("INSERT INTO personas VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",filas)
conexion.commit()#Guarda los datos que modificamos 
conexion.close()#Cierra la base de datos

#Creamos una funcion para sacar la informacion de la DB
def infomracion_db(ruta="npc.db"):
    conexion = sqlite3.connect(ruta)#Abrimos el archivo
    cursor = conexion.cursor()#Nos permite enviar comandos SQL

    #Hacedemos a la tabla personas
    cursor.execute("""
        SELECT falta_opinion, mismas_frases, sigue_tendencias,
               respuestas_automaticas, pasatiempos_genericos,
               siempre_fondo, lenguaje_avanzado,
               aparece_necesitas, categoria
        FROM personas
    """)

###Aqui es donde trasformamos las filas de la DB en una matriz numerica

    datos = np.array(cursor.fetchall(), dtype=float)
    conexion.close()#Cierra la base de datos

    X = datos[:, :8]#Guardamos los inputs
    y = datos[:, 8].astype(int)#Guerdamos la respuesta(0,1,2)

    # Mezclamos los datos para que la red no aprenda un orden especifico
    idx = np.random.permutation(len(X))

    # Partimos los datos: 60% estudiar, 20% validar y 20% examen final
    c1 = int(len(X) * 0.6)
    c2 = int(len(X) * 0.8)

    X_tr, y_tr = X[idx[:c1]], y[idx[:c1]]#El 60% para el entrenamiento
    X_va, y_va = X[idx[c1:c2]], y[idx[c1:c2]]#El 20% para la validación
    X_te, y_te = X[idx[c2:]], y[idx[c2:]]#El 20% final es para la prueba

    #Calculasmos el promedio de cada columna
    imputacion = np.nanmean(X_tr, axis=0)
    
    #Si hay valores (Null), los rellenas con el promedio que sacamos(imputacion)
    X_tr = np.where(np.isnan(X_tr), imputacion, X_tr)
    X_va = np.where(np.isnan(X_va), imputacion, X_va)
    X_te = np.where(np.isnan(X_te), imputacion, X_te)

    #Sacamos el promedio de cada columna del grupo de entrenamiento
    mu = X_tr.mean(axis=0)
    #Mide que tan variadas o desordenados estan los numeros en cada columna
    sd = X_tr.std(axis=0)
    sd[sd == 0] = 1  # Evita que el código explote si dividimos entre cero
    
    #Aplicamos el Z-score: restamos el promedio y dividimos entre el desorden
    #Esto aplasta los numeros grandes y estira los chicos para que todos midan lo mismo
    X_tr = (X_tr - mu) / sd
    X_va = (X_va - mu) / sd
    X_te = (X_te - mu) / sd

    # La red tiene 3 salidas(0,1,2) pero no entiende numeros sueltos, necesita listas
    # np.eye convierte el numero en una lista de tres posiciones con un 1 donde corresonde
    y_tr = np.eye(3)[y_tr]#si el numero es 0 enciendo el primer boton [1,0,0]
    y_va = np.eye(3)[y_va]#Si el numero es 1 [0,1,0]
    y_te = np.eye(3)[y_te]#Si es 2 [0,0,1]
    #Devuelve los datos limpios y acomodados para que la red empiece a entrenar
    return X_tr, y_tr, X_va, y_va, X_te, y_te, imputacion, mu, sd


X_tr, y_tr, X_va, y_va, X_te, y_te, imputacion, mu, sd = infomracion_db()

#funcioes de activacion
def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def softmax(z):
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

# Revisa que tan lejos se quedo la red del resultado real (Multiclase)
def cross_entropy(y_real, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(np.sum(y_real * np.log(y_pred), axis=1))

#Creamos una clase Red Neuronal
class RedNeuronal:
    def __init__(self):
        self.capas = []#Aqui se guardan todas als capas

    #Nos permite agregar las capas ala red y inicializamos los pesos con He
    def add_layer(self, n_in, n_out, activacion="relu"):
        if activacion == "relu":
            w = np.random.randn(n_out, n_in) * np.sqrt(2 / n_in)
        else:
            w = np.random.randn(n_out, n_in) * np.sqrt(1 / n_in)

        b = np.zeros(n_out)
        #Guarda la informacion de la capa
        self.capas.append({"w": w,"b": b,"activacion": activacion})

    #Propagacion hacia adelante
    def fd(self, x):
        self.cache = []#Se guardan los datos para el aprendizaje
        a = x#La entrada comienza siendo x
        for capa in self.capas:
            a_in = a#Guarda la entrada de esta capa
            #Aqui se hacen las operacion de multiplicaion y activacion
            z = capa["w"] @ a + capa["b"]
            if capa["activacion"] == "relu":
                a = relu(z)
            else:
                a = softmax(z)
            #Guarda los resultados obtenidos
            self.cache.append({"z": z,"a": a,"a_in": a_in})
        #Devuleve la predicion de la red
        return a

    #Propagacion hacia atras
    def bk(self, y, alpha):
        grads = []#Se guardan los gradientes
        delta = self.cache[-1]["a"] - y#Empezamos desde la ultima capa
        #Recorre las capas de atras hacia adelante
        for i in reversed(range(len(self.capas))):
            capa = self.capas[i]
            a_in = self.cache[i]["a_in"]
            #Se calculacuanto debe cambiar cada peso y cada bias
            dw = np.outer(delta, a_in)
            db = delta.copy()
            #Guarda los gradientes
            grads.insert(0, {"dw": dw, "db": db})
            #Si no estamos en la primera capa pasamos el error hacia atras
            if i > 0:
                da = capa["w"].T @ delta
                #Recuperamos el z de la capa anterior
                z_anterior = self.cache[i - 1]["z"]
                delta = da * relu_deriv(z_anterior)
        #Acualiza los pesos y bias
        for capa, g in zip(self.capas, grads):
            capa["w"] -= alpha * g["dw"]
            capa["b"] -= alpha * g["db"]

# Arquitectura de la red 8,12,10,3
rn = RedNeuronal()
rn.add_layer(8, 12, "relu")
rn.add_layer(12, 10, "relu")
rn.add_layer(10, 3, "softmax")

#Entrenamiento y correcion de la red

epocas = 100
alpha_inicial = 0.05  # Velocidad inicial a la que aprende la red

#Listas vacias para guardar como evoluciona la red en cada epoca
historial_loss = []
historial_val = []

# El bucle principal de aprendizaje
for epoca in range(epocas):
    # El alpha decae un poco en cada época para que la red no "rebote" y aprenda bien
    alpha_actual = alpha_inicial / (1 + 0.05 * epoca)
    # Mezclamos los datos en cada época
    orden = np.random.permutation(len(X_tr))

    for i in orden:
        #La red hace su predicion y luego corrige sus errores personas por persona
        pred = rn.fd(X_tr[i])
        rn.bk(y_tr[i], alpha_actual)

    #Medimos el estado real en cada epoca calculando las perdidas
    pred_tr = np.array([rn.fd(x) for x in X_tr])
    pred_va = np.array([rn.fd(x) for x in X_va])
    #Calculamos que tan lejos o cerca se quedó la red de las respuestas reales en ambos grupos
    perdida_tr = cross_entropy(y_tr, pred_tr)
    perdida_va = cross_entropy(y_va, pred_va)
    #Guardamos los resultados en el historial para graficar
    historial_loss.append(perdida_tr)
    historial_val.append(perdida_va)
    
    #Imprime el progeso cada 10 vueltas
    if (epoca + 1) % 10 == 0:
        print(f"Época {epoca + 1:3d} | Train Loss: {perdida_tr:.4f} | Val Loss: {perdida_va:.4f}")

#Prueva yo

falta_op = 5; frases = 6
modas = 3; auto = 2
hobbies = 1; fondo = 6
sarcasmo = 2; favores = 8

persona_ejemplo = np.array([falta_op, frases, modas, auto, hobbies, fondo, sarcasmo, favores], dtype=float)

#Aplicamos la misma imputación y Z-score que aprendimos de los datos de entrenamiento
persona_ejemplo = np.where(np.isnan(persona_ejemplo), imputacion, persona_ejemplo)
persona_ejemplo = (persona_ejemplo - mu) / sd

prediccion = rn.fd(persona_ejemplo)

categorias = ["Es un NPC","Se esta convirtiendo en NPC","No es un NPC"]
indice = np.argmax(prediccion)

print("\n Predicion yo")
for nombre, prob in zip(categorias, prediccion):
    print(f"{nombre}: {prob * 100:.2f}%")

#Grafica
plt.plot(range(1, epocas + 1), historial_loss, label='Train Loss (Pérdida Entrenamiento)', color='blue')
plt.plot(range(1, epocas + 1), historial_val, label='Val Loss (Pérdida Validación)', color='orange')
plt.title('Historial de Pérdida (Loss) vs Validación durante el Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (Cross Entropy)')
plt.grid(True)
plt.legend()
plt.show()

#Grafica 2
fig, ax = plt.subplots(figsize=(16, 9))

input_nodes = [(1, y) for y in np.linspace(1, 8, 8)]
hidden1_nodes = [(4, y) for y in np.linspace(0.5, 8.5, 12)]
hidden2_nodes = [(7, y) for y in np.linspace(0.8, 8.2, 10)]
output_nodes = [(10, y) for y in [2.5, 4.5, 6.5]]

for i_node in input_nodes:
    for h_node in hidden1_nodes:
        ax.plot([i_node[0], h_node[0]],[i_node[1], h_node[1]],color="gray", alpha=0.25, zorder=1)
for h1_node in hidden1_nodes:
    for h2_node in hidden2_nodes:
        ax.plot([h1_node[0], h2_node[0]],[h1_node[1], h2_node[1]],color="gray", alpha=0.2, zorder=1)
for h2_node in hidden2_nodes:
    for out_node in output_nodes:
        ax.plot([h2_node[0], out_node[0]],[h2_node[1], out_node[1]],color="gray", alpha=0.3, zorder=1)
for node in input_nodes:
    ax.scatter(*node, s=650, color="skyblue", edgecolors="black", linewidths=1, zorder=2)
for node in hidden1_nodes:
    ax.scatter(*node, s=450, color="lightgreen",edgecolors="black", linewidths=1, zorder=2)
for node in hidden2_nodes:
    ax.scatter(*node, s=500, color="orange",edgecolors="black", linewidths=1, zorder=2)
for node in output_nodes:
    ax.scatter(*node, s=1000, color="salmon",edgecolors="black", linewidths=1.5, zorder=2)

nombres = [
    "Falta de opinion propia",
    "Mismas frases de siempre",
    "Sigue tendencias",
    "Respuestas automaticas",
    "Pasatiempos genericos",
    "Siempre esta de fondo",
    "No entiende sarcasmo/ironia",
    "Solo aparece cuando lo necesitas"
]

for i, nombre in enumerate(nombres):
    ax.text(0.55, input_nodes[i][1], nombre,ha="right", va="center",fontsize=9, fontweight="bold")

ax.text(1, 9, "Entrada\n8", ha="center",fontsize=12, fontweight="bold")
ax.text(4, 9, "Oculta 1 (ReLU)\n12", ha="center",fontsize=12, fontweight="bold")
ax.text(7, 9, "Oculta 2 (ReLU)\n10", ha="center",fontsize=12, fontweight="bold")
ax.text(10, 8, "Salida (Softmax)\n3", ha="center",fontsize=12, fontweight="bold")

salidas = ["Es un NPC","Se convierte en NPC","No es un NPC"]

for i, nombre in enumerate(salidas):
    ax.text(11, output_nodes[i][1], nombre,ha="left", va="center", fontsize=10,fontweight="bold")
    
ax.set_xlim(-4.2, 14)
ax.set_ylim(0, 9.8)
ax.axis("off")

plt.title("Red neuronal clasificadora de NPCS (8 → 12 → 10 → 3)",fontsize=15, fontweight="bold", pad=20)
plt.tight_layout()
plt.show()