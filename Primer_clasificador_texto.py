import sqlite3
import numpy as np
import matplotlib.pyplot as plt

'''Red neuronal que predice a que categoria pertenece una pregunta'''
#Conectamos la base de datos con la RedNeuronal
"""Esta red toma texto como entrada y lo normaliza para convertir las 
palabras en una lista de ceros y unos, como es texto la red tiene 367 
entradas lo que significa que hay 367 palabras unicas en nuestro datos."""

"""Imagina que en una pantalla con 367 botones, donde cada boton es una 
palabra de nuestro diccionario, cuando llega una pregunta las luces de 
las palabras que que estan en la oracion se quedan encendidas las demas 
se apagan. La red no lee texto solo mira que convinacion de luces se encendio"""


#La ruta donde esta almacena la base de datos
RUTA = "file:///C:/Users/LENOVO/Documents/sql/Clasificador_texto.db?mode=rw"
OCULTAS = [16]#Capas ocultas
EPOCAS = 400#Vueltas para la aprender
LR = 0.2#Que tan rapido la red ajusta los errores

#Creamos una funcion para poder conectra con la base de datos
def cargar_filas(ruta=RUTA):
    con = sqlite3.connect(ruta, uri=True)#Abrimos la conexion
    cur = con.cursor()#Nos permite enviarle comandos sql
    #Traemos el texto y categoria de la tabla preguntas
    cur.execute("SELECT texto, categoria FROM preguntas")
    filas = cur.fetchall()#Guardamos todas las fila de la lista
    con.close()
    return filas#Devolvemos la lista sin normalizar

#Funcion para normalizar el texto
def limpiar(texto):
    texto = texto.lower()#Pasamos todo a minusculas
    #Quitamos los signos pra que no confunda a la red
    for s in ["¿", "?", ",", "."]:
        texto = texto.replace(s, "")
    #Dividimos el texto en una lista de palabras sueltas
    return texto.split()

#Funcion para convertir el texto en una matriz para que la red pueda utilizar 
def texto_a_vector(texto, idx):
    v = np.zeros(len(idx))#Creamos una fila llena de ceros 
    #Revisamos palabra por palabra
    for p in limpiar(texto):
        if p in idx:
            #Si la palabra existe en nuestro diccionario, ponemos un 1
            v[idx[p]] = 1
    return v#Devolvemos la fila convertida en ceros y unos

"""La red necesita esta función porque no entdiente palabras, solo matrices de numeros
lo que hace es qeu toma texto crudo y devuelve los datos ya normalizados y en una 
matriz de ceros y unos"""
def preparar_datos(filas, semilla=0):
    #Separa texto en un lado y categorias por otro
    textos = [f[0] for f in filas]
    etiq = [f[1] for f in filas]
    palabras = []
    #Juntamos todas las palabras de todas las preguntas en una sola lista gigante
    for t in textos:
        palabras = palabras + limpiar(t)
    #Elimina las palabras duplicadas,y ordena en orden alfabetico
    vocab = sorted(set(palabras))
    #Creamos un diccionario para saber que numero de casillero le toca a cada palabra
    idx = {}
    for i, p in enumerate(vocab):
        idx[p] = i
    #Obtenemos las 4 unicas categorias
    cats = sorted(set(etiq))
    #Inicializamos las matrices de entrada y salida
    X = np.zeros((len(textos), len(vocab)))#Llenas de 1
    Y = np.zeros((len(textos), len(cats)))#Llenas de 0
    #Llenamos las matrices traduciendo de texto a numero cada 
    for i, t in enumerate(textos):
        X[i] = texto_a_vector(t, idx)
        Y[i, cats.index(etiq[i])] = 1
    #Hace que cada vez que ejecutemos sean los mismo datos
    np.random.seed(semilla)
    tr, va, te = [], [], []
    #Separamos los datos asegurando que queden repartidos de forma justa para cada categoria
    for c in range(len(cats)):
        #Buscamos en que lineas estan las preguntas de la categoria que le toca
        pos = np.where(Y[:, c] == 1)[0]
        #Mesclamos para que no tengan orden fijo
        pos = np.random.permutation(pos)
        #Calculamos la divicion de datos 60 - 20 - 20
        a = int(len(pos) * 0.6)
        b = int(len(pos) * 0.8)
        #Guardamos los bloques de datos para las diferentes etapas del entrenamiento
        tr = tr + list(pos[:a])#60% para entrenar
        va = va + list(pos[a:b])#20% Para validar el entrenamiento
        te = te + list(pos[b:])#20% para tomar el examen final
    #Devulve dos datos segmentados para la red neuronal
    return {
        "Xtr": X[tr], "Ytr": Y[tr],#Datos entrenamiento
        "Xva": X[va], "Yva": Y[va],#Datos validacion
        "Xte": X[te], "Yte": Y[te],#Datos prueba final
        #Los diccionarios y nombres de las categorias
        "vocab": vocab, "idx": idx, "cats": cats,
    }


#Funciones para el entrenamiento en RedNeuronal
#Funcion de activacion multicalse
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

#Calcula que tan grande fue el erro de la red
def perdida(P, Y):
    return -np.mean(np.sum(Y * np.log(P + 1e-9), axis=1))

#Crea una tabal cuadrada que compara los datos contra la prediccion de la red
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
        self.lr = lr#Guardamos la velocidad ala que aprende la red
        self.W = []#Guarda los pesos 
        self.b = []#Guarda las bias
        #Conecta las capas de principio a fin
        for i in range(len(capas) - 1):
            n_in = capas[i]#Neuronas de entrada
            n_out = capas[i + 1]#Neuronas de salida
            #Normalizamos con 'He', y llenamos los pesos con numeros aleatorios
            self.W.append(np.random.randn(n_in, n_out) * np.sqrt(2 / n_in))
            self.b.append(np.zeros(n_out))#Todas las bias empienzan en cero
        
    #Propagacion hacia adelante
    def fd(self, X):
        self.A = [X]#Valores de entrada
        self.Z = []
        #Viajamos caap por cada hacia adelante
        for i in range(len(self.W)):
            z = self.A[-1] @ self.W[i] + self.b[i]# w1 * x + b1
            self.Z.append(z)
            if i == len(self.W) - 1:
                a = softmax(z)#Activamos con softmax
            else:
                a = np.maximum(0, z)#si es una capa intermedia activamos con relu
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
            db = dz.sum(axis=0)#Calculas cuanto debe cambiar las bias
            if i > 0:
                dz_ant = (dz @ self.W[i].T) * (self.Z[i - 1] > 0)
            #Corregimos los pesos y bias
            self.W[i] -= self.lr * dW
            self.b[i] -= self.lr * db
            if i > 0:
                dz = dz_ant
    #Recibe una pregunta y devuelve el numero de la pregunta
    def predecir(self, X):
        return self.fd(X).argmax(axis=1)#Cual fue la categoria con la probabilidad mas alta
    
    #Es el bucle principal, donde entrena y se corrigue en varias vueltas
    def entrenar(self, Xtr, Ytr, Xva, Yva, epocas=300):
        C = Ytr.shape[1]#Obtenemos el numero total de las categorias 
        #Guardamos los resultados para graficar el los, val, f1 
        h = {"loss_tr": [], "loss_va": [], "f1_tr": [], "f1_va": []}
        for ep in range(epocas):
            #La red comienza el entrenamiento con los datos del entrenamiento
            P = self.fd(Xtr)
            h["loss_tr"].append(perdida(P, Ytr))#Medimos que tan grande fue el error
            h["f1_tr"].append(f1_por_clase(matriz_confusion(Ytr.argmax(axis=1), P.argmax(axis=1), C)).mean())
            #La red comienza a correguir los pesos y bias
            self.bk(Ytr)
            #Evaluamos a la red con preguntas que no vio en el entrenamiento
            Pv = self.fd(Xva)
            h["loss_va"].append(perdida(Pv, Yva))
            h["f1_va"].append(f1_por_clase(matriz_confusion(Yva.argmax(axis=1), Pv.argmax(axis=1), C)).mean())
            #Imprimimos los resultado cada 50 vueltas para comprovar el progreso
            if ep % 50 == 0 or ep == epocas - 1:
                print(f"época {ep:3d} | loss tr {h['loss_tr'][-1]:.3f} va {h['loss_va'][-1]:.3f} | F1 tr {h['f1_tr'][-1]:.3f} va {h['f1_va'][-1]:.3f}")
        return h #Devolvemos el historial completo para poder graficar las curvas de aprendizaje


#Aqui comienza a ejectuarse todo el programa, se llama al objeto RedNeuronal
if __name__ == "__main__":
    np.random.seed(0)#Para que siempre sea el mismo orden 
    #Traemos las preguntas de las base de datos para lipiarlas y dividirlas
    d = preparar_datos(cargar_filas())
    cats = d["cats"]#Lista de las 4 categorias
    print("Palabras:", len(d["vocab"]), "| train:", len(d["Xtr"]), "| val:", len(d["Xva"]), "| test:", len(d["Xte"]))
    #Diseña la arquitectura de la red
    capas = [len(d["vocab"])] + OCULTAS + [len(cats)]
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
    plt.savefig("curvas.png", dpi=150)
    plt.show()
    
    #Guardamos las respuestas del examen final
    real = d["Yte"].argmax(axis=1)
    pred = red.predecir(d["Xte"])#La red responde para ver que tanto aprendio
    #Crea una tabla apra ver en cuales acerto y cuales se confundio
    M = matriz_confusion(real, pred, len(cats))
    f1 = f1_por_clase(M)#Saca la calificaion
    #Porcentaje de respuestas correctas
    print("\nTEST -> precisión:", round(np.mean(real == pred), 3), "| F1 promedio:", round(f1.mean(), 3))
    for i, c in enumerate(cats):
        print(f"  {c:15s} F1 = {f1[i]:.3f}")
    
    
    print("\n Que en cuales categorias se equivoca mas (Programación)")
    print(cats)
    print(M.astype(int))

    print("PRUEBAS CON EJEMPLOS NUEVOS")
    print("----------------------")
    
    #Ponemos a pruba a la red con preguntas que nunca a visto que nosotros se las pasamos
    frases_prueba = [
        "¿Que hace una red neuronal?",
        "¿Como se hacede a un elemento especifico de una lista?",
        "Que pasria si no tenemos suficientes datos",
        "¿Ella me ama?"
    ]
    
    #Limpia los datos y entrena a la red
    for frase in frases_prueba:
        #Convertimos las frases nuevas en una listas de 0, 1
        vec = texto_a_vector(frase, d["idx"]).reshape(1, -1)
        probabilidades = red.fd(vec)[0]#Porcentaje de cada categoria
        idx_ganador = probabilidades.argmax()#Buscamos cual categoria tiene el porcentaje mas alto
        categoria_asignada = cats[idx_ganador]
        confianza = probabilidades[idx_ganador] * 100#Calculas que tanta seguridad tiene 
        print(f"Frase: '{frase}'")
        print(f" Categoría mas probable: {categoria_asignada} (Confianza: {confianza:.1f}%)")
        print("-" * 50)
        
        
    #Como se veria la red graficamente, el numero de entradas es de 367 pero no alcanza 
    fig, ax = plt.subplots(figsize=(16, 9))
    n_entrada = len(d["vocab"])
    n_oculta = OCULTAS[0]
    n_salida = len(cats)
    input_nodes = [(1, y) for y in np.linspace(1, 8, min(n_entrada, 10))]
    hidden1_nodes = [(5, y) for y in np.linspace(0.5, 8.5, n_oculta)]
    output_nodes = [(9, y) for y in np.linspace(2, 7, n_salida)]
    for i_node in input_nodes:
        for h_node in hidden1_nodes:
            ax.plot([i_node[0], h_node[0]], [i_node[1], h_node[1]], color="gray", alpha=0.15, zorder=1)
    for h_node in hidden1_nodes:
        for out_node in output_nodes:
            ax.plot([h_node[0], out_node[0]], [h_node[1], out_node[1]], color="gray", alpha=0.25, zorder=1)
    for node in input_nodes:
        ax.scatter(*node, s=400, color="skyblue", edgecolors="black", linewidths=1, zorder=2)
    for node in hidden1_nodes:
        ax.scatter(*node, s=450, color="lightgreen", edgecolors="black", linewidths=1, zorder=2)
    for node in output_nodes:
        ax.scatter(*node, s=800, color="salmon", edgecolors="black", linewidths=1.5, zorder=2)
    ax.text(1, 8.8, f"Entrada\n({n_entrada} palabras)", ha="center", fontsize=11, fontweight="bold")
    ax.text(5, 8.8, f"Oculta 1 (ReLU)\n({n_oculta} neuronas)", ha="center", fontsize=11, fontweight="bold")
    ax.text(9, 8.8, f"Salida (Softmax)\n({n_salida} categorías)", ha="center", fontsize=11, fontweight="bold")
    for i, nombre in enumerate(cats):
        ax.text(9.5, output_nodes[i][1], nombre, ha="left", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(-1, 13)
    ax.set_ylim(0, 9.8)
    ax.axis("off")
    plt.title(f"Red Neuronal Clasificadora de Textos ({n_entrada} → {n_oculta} → {n_salida})", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.show()