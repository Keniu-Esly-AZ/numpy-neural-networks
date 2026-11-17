import csv
import os
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

'''Red neuronal que predice la especie de un pingüino por 4 entradas'''
"Entredas"
#Largo de pico
#Ancho pico
#Largo aleta
#Peso

"""Esta red toma 4 medidas de un pingüino (largo del pico, ancho del pico,
largo de la aleta y peso) y decide si es Adelie, Chinstrap o Gentoo. A diferencia
de la red de texto, aqui no hay palabras, las entradas ya son numeros, pero hay
que escalarlos para que el peso (miles de gramos) no tape a las demas medidas."""

#La carpeta donde esta este script, ahi se guarda la base de datos
CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA_DB = os.path.join(CARPETA, "penguins.db")#La ruta donde se crea la base sqlite3
OCULTAS = [8]#Capas ocultas
EPOCAS = 500#Vueltas para aprender
LR = 0.1#Que tan rapido la red ajusta los errores


"""Los datos se descargaron como archivo cvs por
eso se pasa a formato db"""
def crear_base():
    if os.path.exists(RUTA_DB):#Si la base ya existe no la volvemos a crear
        return
    ruta_csv = None
    #Buscamos el archivo por esos dos nombres
    for nombre in ["penguins.csv", "penguins.csv.txt"]:
        candidato = os.path.join(CARPETA, nombre)
        if os.path.exists(candidato):
            ruta_csv = candidato
    if ruta_csv is None:
        raise FileNotFoundError("Pon penguins.csv junto a este script")
    #Creamos la tabal con sus campos y sus registros
    con = sqlite3.connect(RUTA_DB)#Abrimos la conexion
    cur = con.cursor()#Nos permite enviarle comandos sql
    cur.execute("""CREATE TABLE pinguinos (
        id INTEGER PRIMARY KEY,
        especie TEXT, isla TEXT,
        largo_pico REAL, ancho_pico REAL,
        largo_aleta REAL, peso REAL,
        sexo TEXT)""")
    #Editamos el archivo cvs para transformarlo en campos y registros
    with open(ruta_csv, newline="", encoding="utf-8") as archivo:
        lector = csv.reader(archivo)
        next(lector)#Saltamos la primera fila porque son los nombres de las columnas
        for fila in lector:
            #Las celdas vacias se guardan como NULL
            valores = [v if v != "" else None for v in fila]
            cur.execute("""INSERT INTO pinguinos
                (especie, isla, largo_pico, ancho_pico, largo_aleta, peso, sexo)
                VALUES (?, ?, ?, ?, ?, ?, ?)""", valores)
    con.commit()#Guardamos los cambios en el archivo, sin esto la base queda vacia
    con.close()


#Creamos una funcion para poder conectar con la base de datos
def cargar_filas():
    con = sqlite3.connect(RUTA_DB)#Abrimos la conexion
    cur = con.cursor()
    #Traemos la especie y las 4 medidas
    cur.execute("""SELECT especie, largo_pico, ancho_pico, largo_aleta, peso
                   FROM pinguinos
                   WHERE largo_pico IS NOT NULL AND ancho_pico IS NOT NULL
                     AND largo_aleta IS NOT NULL AND peso IS NOT NULL""")
    filas = cur.fetchall()#Guardamos las filas en una lista
    con.close()
    return filas


"""La red necesita esta funcion porque solo entiende numeros pequeños y parecidos entre si.
Toma las filas de la base de datos, escala las medidas, convierte las especies en
ceros y unos y reparte los datos en entrenamiento, validacion y prueba."""
def preparar_datos(filas, semilla=0):
    etiq = [f[0] for f in filas]#Separamos las especies (texto)
    X = np.array([f[1:] for f in filas], dtype=float)#Las 4 medidas de cada pinguino
    cats = sorted(set(etiq))#Obtenemos las 3 especies unicas en orden alfabetico
    Y = np.zeros((len(filas), len(cats)))#Matriz de salida llena de 0
    #Marcamos con un 1 la especie de cada pinguino, Adelie seria [1, 0, 0]
    for i, e in enumerate(etiq):
        Y[i, cats.index(e)] = 1

    #Hace que cada vez que ejecutemos sean los mismos datos
    np.random.seed(semilla)
    tr, va, te = [], [], []
    #Separamos los datos asegurando que queden repartidos de forma justa para cada especie
    for c in range(len(cats)):
        #Buscamos en que lineas estan los pinguinos de la especie que le toca
        pos = np.where(Y[:, c] == 1)[0]
        #Mezclamos para que no tengan orden fijo
        pos = np.random.permutation(pos)
        #Calculamos la division de datos 60 - 20 - 20
        a = int(len(pos) * 0.6)
        b = int(len(pos) * 0.8)
        tr = tr + list(pos[:a])#60% para entrenar
        va = va + list(pos[a:b])#20% para validar el entrenamiento
        te = te + list(pos[b:])#20% para tomar el examen final

    #Escalamos las medidas: a cada una le restamos su promedio y la dividimos por su desviacion
    #Los calculos se hacen solo con los datos de entrenamiento para que el examen final sea justo
    media = X[tr].mean(axis=0)
    desv = X[tr].std(axis=0)
    X = (X - media) / desv

    #Devuelve los datos segmentados para la red neuronal
    return {
        "Xtr": X[tr], "Ytr": Y[tr],#Datos entrenamiento
        "Xva": X[va], "Yva": Y[va],#Datos validacion
        "Xte": X[te], "Yte": Y[te],#Datos prueba final
        #La media y desviacion sirven para escalar pinguinos nuevos igual que los de entrenamiento
        "media": media, "desv": desv, "cats": cats,
    }


#Funciones para el entrenamiento en RedNeuronal
#Funcion de activacion multiclase
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

#Calcula que tan grande fue el error de la red
def perdida(P, Y):
    return -np.mean(np.sum(Y * np.log(P + 1e-9), axis=1))

#Crea una tabla cuadrada que compara los datos contra la prediccion de la red
def matriz_confusion(real, pred, C):
    M = np.zeros((C, C))
    for r, p in zip(real, pred):
        M[r, p] += 1
    return M

#Evalua el rendimiento de la red con nota de 0 a 1
def f1_por_clase(M):
    f1 = np.zeros(len(M))
    for c in range(len(M)):
        tp = M[c, c]
        fp = M[:, c].sum() - tp
        fn = M[c, :].sum() - tp
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
    return f1

#Es donde la red aprende, ajusta y devuelve
class RedNeuronal:
    def __init__(self, capas, lr=0.5):
        self.lr = lr#Guardamos la velocidad a la que aprende la red
        self.W = []#Guarda los pesos
        self.b = []#Guarda las bias
        #Conecta las capas de principio a fin
        for i in range(len(capas) - 1):
            n_in = capas[i]#Neuronas de entrada
            n_out = capas[i + 1]#Neuronas de salida
            #Normalizamos con 'He', y llenamos los pesos con numeros aleatorios
            self.W.append(np.random.randn(n_in, n_out) * np.sqrt(2 / n_in))
            self.b.append(np.zeros(n_out))#Todas las bias empiezan en cero

    #Propagacion hacia adelante
    def fd(self, X):
        self.A = [X]#Valores de entrada
        self.Z = []
        #Viajamos capa por capa hacia adelante
        for i in range(len(self.W)):
            z = self.A[-1] @ self.W[i] + self.b[i]# w1 * x + b1
            self.Z.append(z)
            if i == len(self.W) - 1:
                a = softmax(z)#Activamos con softmax
            else:
                a = np.maximum(0, z)#Si es una capa intermedia activamos con relu
            self.A.append(a)
        return self.A[-1]#Devolvemos la prediccion final de la red

    #Propagacion hacia atras
    def bk(self, Y):
        n = len(Y)
        #Calculamos la falla inicial
        dz = (self.A[-1] - Y) / n
        #Comenzamos desde la ultima capa hasta la primera
        for i in reversed(range(len(self.W))):
            dW = self.A[i].T @ dz#Calculamos cuanto deben cambiar los pesos
            db = dz.sum(axis=0)#Calculamos cuanto deben cambiar las bias
            if i > 0:
                dz_ant = (dz @ self.W[i].T) * (self.Z[i - 1] > 0)
            #Corregimos los pesos y bias
            self.W[i] -= self.lr * dW
            self.b[i] -= self.lr * db
            if i > 0:
                dz = dz_ant

    #Recibe los pinguinos y devuelve el numero de la especie
    def predecir(self, X):
        return self.fd(X).argmax(axis=1)#Cual fue la especie con la probabilidad mas alta

    #Es el bucle principal, donde entrena y se corrige en varias vueltas
    def entrenar(self, Xtr, Ytr, Xva, Yva, epocas=300):
        C = Ytr.shape[1]#Obtenemos el numero total de especies
        #Guardamos los resultados para graficar loss y f1
        h = {"loss_tr": [], "loss_va": [], "f1_tr": [], "f1_va": []}
        for ep in range(epocas):
            #La red comienza el entrenamiento con los datos del entrenamiento
            P = self.fd(Xtr)
            h["loss_tr"].append(perdida(P, Ytr))#Medimos que tan grande fue el error
            h["f1_tr"].append(f1_por_clase(matriz_confusion(Ytr.argmax(axis=1), P.argmax(axis=1), C)).mean())
            #La red comienza a corregir los pesos y bias
            self.bk(Ytr)
            #Evaluamos a la red con pinguinos que no vio en el entrenamiento
            Pv = self.fd(Xva)
            h["loss_va"].append(perdida(Pv, Yva))
            h["f1_va"].append(f1_por_clase(matriz_confusion(Yva.argmax(axis=1), Pv.argmax(axis=1), C)).mean())
            #Imprimimos los resultados cada 100 vueltas para comprobar el progreso
            if ep % 100 == 0 or ep == epocas - 1:
                print(f"época {ep:3d} | loss tr {h['loss_tr'][-1]:.3f} va {h['loss_va'][-1]:.3f} | F1 tr {h['f1_tr'][-1]:.3f} va {h['f1_va'][-1]:.3f}")
        return h#Devolvemos el historial completo para poder graficar las curvas de aprendizaje


#Aqui comienza a ejecutarse todo el programa, se llama al objeto RedNeuronal
if __name__ == "__main__":
    np.random.seed(0)#Para que siempre sea el mismo orden
    crear_base()#Si es la primera vez, pasa el csv a sqlite3
    #Traemos los pinguinos de la base de datos para escalarlos y dividirlos
    d = preparar_datos(cargar_filas())
    cats = d["cats"]#Lista de las 3 especies
    print("train:", len(d["Xtr"]), "| val:", len(d["Xva"]), "| test:", len(d["Xte"]))
    #Disena la arquitectura de la red: 4 entradas, 8 ocultas, 3 salidas
    capas = [4] + OCULTAS + [len(cats)]
    red = RedNeuronal(capas, lr=LR)#Llamamos al objeto RedNeuronal
    #Comienza el entrenamiento
    h = red.entrenar(d["Xtr"], d["Ytr"], d["Xva"], d["Yva"], EPOCAS)


    #Graficas del entrenamiento
    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.plot(h["loss_tr"], label="loss entrenamiento")
    plt.plot(h["loss_va"], label="loss validación")
    plt.xlabel("época")
    plt.ylabel("loss")
    plt.title("Pérdida")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(h["f1_tr"], label="F1 entrenamiento")
    plt.plot(h["f1_va"], label="F1 validación")
    plt.ylim(0, 1.05)
    plt.xlabel("época")
    plt.ylabel("F1")
    plt.title("F1")
    plt.legend()
    plt.tight_layout()
    plt.show()

    #Guardamos las respuestas del examen final
    real = d["Yte"].argmax(axis=1)
    pred = red.predecir(d["Xte"])#La red responde para ver que tanto aprendio
    #Crea una tabla para ver en cuales acerto y en cuales se confundio
    M = matriz_confusion(real, pred, len(cats))
    f1 = f1_por_clase(M)#Saca la calificacion
    #Porcentaje de respuestas correctas
    print("\nTEST precisión:", round(np.mean(real == pred), 3), "| F1 promedio:", round(f1.mean(), 3))
    for i, c in enumerate(cats):
        print(f"  {c:10s} F1 = {f1[i]:.3f}")

    print("\nEn cuales especies se equivoca mas (filas = real, columnas = predicho)")
    print(cats)
    print(M.astype(int))

    print("\nPruevas de pinguins nuevos(Datos sinteticos)")
    print("----------------------")

    #Ponemos a prueba a la red con pinguinos que nunca ha visto, medidas en este orden:
    #largo pico(mm), ancho pico(mm), largo aleta(mm), peso(g)
    pruebas = [
        ("Pingüino A", [38.5, 18.2, 190, 3650]),
        ("Pingüino B", [49.5, 18.5, 197, 3700]),
        ("Pingüino C", [47.0, 14.8, 216, 5100]),
    ]

    #Escala los datos y le pide a la red la especie de cada pinguino
    for nombre, medidas in pruebas:
        #Escalamos con la misma media y desviacion del entrenamiento
        x = (np.array(medidas, dtype=float) - d["media"]) / d["desv"]
        probs = red.fd(x.reshape(1, -1))[0]#Porcentaje de cada especie
        ganador = probs.argmax()#Buscamos cual especie tiene el porcentaje mas alto
        print(f"{nombre}: {medidas}")
        print(f"  Especie mas probable: {cats[ganador]} (Confianza: {probs[ganador] * 100:.1f}%)")#Que tanta seguridad tiene
        print("  Todas:", {c: f"{p * 100:.1f}%" for c, p in zip(cats, probs)})
        print("-" * 50)


    #Como se veria la red graficamente, 4 entradas, 8 neuronas ocultas y 3 salidas
    fig, ax = plt.subplots(figsize=(16, 9))
    medidas_nombres = ["Largo pico", "Ancho pico", "Largo aleta", "Peso"]
    n_entrada = len(medidas_nombres)
    n_oculta = OCULTAS[0]
    n_salida = len(cats)
    input_nodes = [(1, y) for y in np.linspace(7, 2, n_entrada)]
    hidden1_nodes = [(5, y) for y in np.linspace(0.5, 8.5, n_oculta)]
    output_nodes = [(9, y) for y in np.linspace(7, 2, n_salida)]
    for i_node in input_nodes:
        for h_node in hidden1_nodes:
            ax.plot([i_node[0], h_node[0]], [i_node[1], h_node[1]], color="gray", alpha=0.2, zorder=1)
    for h_node in hidden1_nodes:
        for out_node in output_nodes:
            ax.plot([h_node[0], out_node[0]], [h_node[1], out_node[1]], color="gray", alpha=0.25, zorder=1)
    for node in input_nodes:
        ax.scatter(*node, s=600, color="skyblue", edgecolors="black", linewidths=1, zorder=2)
    for node in hidden1_nodes:
        ax.scatter(*node, s=450, color="lightgreen", edgecolors="black", linewidths=1, zorder=2)
    for node in output_nodes:
        ax.scatter(*node, s=800, color="salmon", edgecolors="black", linewidths=1.5, zorder=2)
    ax.text(1, 8.8, f"Entrada\n({n_entrada} medidas)", ha="center", fontsize=11, fontweight="bold")
    ax.text(5, 8.8, f"Oculta 1 (ReLU)\n({n_oculta} neuronas)", ha="center", fontsize=11, fontweight="bold")
    ax.text(9, 8.8, f"Salida (Softmax)\n({n_salida} especies)", ha="center", fontsize=11, fontweight="bold")
    for i, nombre in enumerate(medidas_nombres):
        ax.text(0.6, input_nodes[i][1], nombre, ha="right", va="center", fontsize=10, fontweight="bold")
    for i, nombre in enumerate(cats):
        ax.text(9.5, output_nodes[i][1], nombre, ha="left", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(-2.5, 12)
    ax.set_ylim(0, 9.8)
    ax.axis("off")
    plt.title(f"Red Neuronal Clasificadora de Pingüinos ({n_entrada} → {n_oculta} → {n_salida})", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.show()
    