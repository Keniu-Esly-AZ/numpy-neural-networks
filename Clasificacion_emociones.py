import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import re#Para limpiar texto antes de usarlo
import json#Leer y guardar la configuracion de la red
from collections import Counter#Contar cuantas veces aparece un elemento(vocabulario)
from matplotlib.collections import LineCollection#Dibujar muchas lineas(Grafica)

'''Red neuronal que detecta que emociones expresa una frase'''
"""Esta red toma una frase en ingles y la convierte en una lista de ceros y unos,
como tiene 10000 entradas significa que conoce las 10000 palabras mas usadas"""
"""Utilizamos sigmoid porque una misma frase puede expresar diferentes emociones
a esto se le llama clasificacion multietiqueta, softmax es para clasificacion
multiclase"""

RUTA = r"C:\Users\LENOVO\Documents\python\Neuronas\numpy-neural-networks\goemotions.db"
OCULTAS = [64, 32]#Capas ocultas
EPOCAS = 40#Vueltas para aprender
LR = 0.2#Que tan rapido la red ajusta los errores
BATCH = 256#Cuantas frases ve la red antes de corregirse
MAX_VOCAB = 10000#Maximo de palabras que conoce la red
MIN_FREQ = 2#Una palabra debe aparecer en al menos 2 frases para entrar al diccionario
UMBRAL = 0.5#Decide a partir de que numero la emocion esta en la frase(no es fijo)
SEMILLA = 42#Para que siempre salgan los mismos resultados
PESO_MAX = 10#Tope del peso de las emociones raras
MODO = "entrenar"# entrenar-predecir

#Funcion para traer los datos de la base de datos
def cargar_datos(ruta=RUTA):
    con = sqlite3.connect(ruta)#Abrimos la conexion
    cur = con.cursor()#Nos permite enviarle comandos sql
    #Traemos cada frase con su conjunto (train, dev, test) y sus emociones
    cur.execute("""
        SELECT m.id, m.texto, m.conjunto, me.emocion_id
        FROM muestras m
        LEFT JOIN muestra_emocion me ON m.id = me.muestra_id
        ORDER BY m.id, me.emocion_id
    """)
    filas = cur.fetchall()#Guardamos todas las filas en una lista
    #Traemos los nombres de las emociones
    cur.execute("SELECT id, nombre FROM emociones ORDER BY id")
    emociones = dict(cur.fetchall())#Diccionario {id: nombre}
    con.close()
    datos = {}#Guardamos cada grase una sola vez
    # Una frase con 3 emociones llega en 3 filas, aqui las juntamos en una sola
    for identificador, texto, conjunto, emocion in filas:
        if identificador not in datos:
            datos[identificador] = {"texto": texto, "conjunto": conjunto, "emociones": []}
        if emocion is not None:
            datos[identificador]["emociones"].append(emocion)
    return list(datos.values()), emociones#Devolvemos frases y nombres sin normalizar

#Funcion para normalizar el texto
def limpiar(texto):
    texto = texto.lower()#Pasamos todo a minusculas
    #Dejamos solo palabras, y conservamos las contracciones como "don't"
    return re.findall(r"[a-z]+(?:'[a-z]+)?", texto)

#Funcion que crea el diccionario de palabras
def crear_vocabulario(textos):
    frecuencia = Counter()#Cuenta en cuantas frases aparece cada palabra
    for texto in textos:
        frecuencia.update(set(limpiar(texto)))#set evita contar dos veces en la misma frase
    #Nos quedamos con las palabras que aparecen suficientes veces, de la mas comun a la menos comun
    palabras = [p for p, cantidad in frecuencia.most_common() if cantidad >= MIN_FREQ]
    palabras = palabras[:MAX_VOCAB]#Cortamos al maximo permitido
    idx = {p: i for i, p in enumerate(palabras)}#Le damos a cada palabra el numero de su casila
    return palabras, idx

#Funcion para convertir un grupo de frases en matrices de ceros y unos
def crear_lote(textos, etiquetas, idx, n_clases):
    #Creamos la matriz de entradas, fila por frase, columna por palabra
    X = np.zeros((len(textos), len(idx)), dtype=np.float32)
    #Creamos la matriz de repuesta esperadas. Fila por frase, columna por emocion
    Y = np.zeros((len(textos), n_clases), dtype=np.float32)
    for i, texto in enumerate(textos):
        for palabra in set(texto):#Elimina palabras repetidas dentro de una frase
            j = idx.get(palabra)
            if j is not None:
                X[i, j] = 1#Si la palabra esta en el diccionario encendemos su luz
        for emocion in etiquetas[i]:
            Y[i, emocion] = 1#Encendemos cada emocion que tiene la frase
    return X, Y

"""Calcula un peso para cada emocion segun que tan rara es.
Las emociones poco frecuentes reciben un peso mayor para que la red no las ignore
y el error cuesta mas cuando se equivoque con ellas"""
def calcular_pesos(etiquetas, n_clases):#Resive la lista de emociones de cada frase
    conteo = np.zeros(n_clases)#Cuantas frases tienen cada emocion
    for lista in etiquetas:
        for e in lista:#Recorre cada emocion individual
            conteo[e] += 1
    conteo = np.maximum(conteo, 1)#Evitamos dividir entre cero
    pesos = np.sqrt((len(etiquetas) - conteo) / conteo)#Emocion rara = peso alto, comun = peso bajo
    return np.clip(pesos, 1, PESO_MAX).astype(np.float32)#Limitamos entre 1 y PESO_MAX


#Funcion para separar y preparar todos los datos
def preparar_datos(datos, emociones):
    #Usamos la separacion original de GoEmotions
    train = [d for d in datos if d["conjunto"] == "train"]#60%
    val = [d for d in datos if d["conjunto"] == "dev"]#20%
    test = [d for d in datos if d["conjunto"] == "test"]#20%
    #El diccionario se crea solo con entrenamiento, para que la red no vea el examen
    palabras, idx = crear_vocabulario([d["texto"] for d in train])
    #Pasamos los id originales de las emociones a posiciones 0, 1, 2...
    ids_emociones = sorted(emociones)
    mapa = {emocion: i for i, emocion in enumerate(ids_emociones)}
    nombres = [emociones[i] for i in ids_emociones]
    #Funcion interna que limpia las frases y traduce las emociones de un conjunto
    def extraer(conjunto):
        textos = [limpiar(d["texto"]) for d in conjunto]
        etiquetas = [[mapa[e] for e in d["emociones"]] for d in conjunto]
        return textos, etiquetas
    Xtr, Ytr = extraer(train)
    Xva, Yva = extraer(val)
    Xte, Yte = extraer(test)
    pesos = calcular_pesos(Ytr, len(nombres))#Los pesos se calculan solo con entrenamiento
    return {
        "Xtr": Xtr, "Ytr": Ytr,#Datos entrenamiento
        "Xva": Xva, "Yva": Yva,#Datos validacion
        "Xte": Xte, "Yte": Yte,#Datos prueba final
        #El diccionario, los nombres de las emociones y los pesos
        "vocab": palabras, "idx": idx, "cats": nombres, "pesos": pesos,
    }

#Funciones de activacion
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    z = np.clip(z, -50, 50)#Evitamos numeros gigantes que rompen la exponencial
    return 1 / (1 + np.exp(-z))

"""La perdida ahora recibe los pesos el error se multiplica por el peso
de esa emocion. Asi equivocarse con una emocion rara cuesta mucho mas 
que equivocarse con una emocion comun"""
def perdida(z, Y, pesos):
    sp = np.log1p(np.exp(-np.abs(z)))#Parte comun, estable para numeros grandes
    error_si = np.maximum(-z, 0) + sp#Error cuando la emocion SI estaba
    error_no = np.maximum(z, 0) + sp#Error cuando la emocion NO estaba
    return (pesos * Y * error_si + (1 - Y) * error_no).mean()

#Evalua el rendimiento con nota de 0 a 1, aqui se guardan aciertos y fallos
def calcular_f1(tp, fp, fn):
    #Que tanto puede confiar cuando la red dice "aqui hay alegria"
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    #Que tan buena es para atrapar las emociones reales 
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    #F1 mezcla las dos notas en una sola
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    total = 2 * tp.sum() + fp.sum() + fn.sum()
    micro = 2 * tp.sum() / total if total > 0 else 0#F1 juntando todas las emociones
    return f1.mean(), micro, f1#Devuelve los tres resultados 


#Es donde la red aprende, ajusta y devuelve
class RedNeuronal:
    def __init__(self, capas, lr=0.05):
        self.lr = lr#Guardamos la velocidad a la que aprende la red
        self.W = []#Guarda los pesos
        self.b = []#Guarda los bias
        #Conecta las capas de principio a fin
        for i in range(len(capas) - 1):
            n_in = capas[i]#Neuronas de entrada
            n_out = capas[i + 1]#Neuronas de salida
            #La ultima capa usa escala mas pequena porque su activacion es sigmoid
            escala = np.sqrt(1 / n_in) if i == len(capas) - 2 else np.sqrt(2 / n_in)
            self.W.append((np.random.randn(n_in, n_out) * escala).astype(np.float32))
            self.b.append(np.zeros(n_out, dtype=np.float32))#Todos los bias empiezan en cero

    #Propagacion hacia adelante
    def fd(self, X):
        self.A = [X]#Valores de entrada
        self.Z = []
        #Viajamos capa por capa hacia adelante
        for i in range(len(self.W)):
            z = self.A[-1] @ self.W[i] + self.b[i]# w1 * x + b1
            self.Z.append(z)
            if i == len(self.W) - 1:
                a = sigmoid(z)#La capa final da una probabilidad por emocion
            else:
                a = relu(z)#Las capas intermedias usan relu
            self.A.append(a)
        return self.A[-1]#Devolvemos la prediccion final de la red

    """A diferencai del bk normal, aqui se agrega 'pesos' como elementos extra
    para castigar los fallos en las emociones raras""" 
    #Propagacion hacia atras
    def bk(self, Y, pesos):
        n = len(Y)#Cantidad de frases del lote
        #Error inicial: igual a (prediccion - real) pero con el peso de cada emocion
        dz = (self.A[-1] * (1 + (pesos - 1) * Y) - pesos * Y) / n
        #Comenzamos desde la ultima capa hasta la primera
        for i in reversed(range(len(self.W))):
            dW = self.A[i].T @ dz#Cuanto deben cambiar los pesos
            db = dz.sum(axis=0)#Cuanto deben cambiar los bias
            if i > 0:
                dz_ant = (dz @ self.W[i].T) * (self.Z[i - 1] > 0)
            #Corregimos los pesos y bias
            self.W[i] -= self.lr * dW
            self.b[i] -= self.lr * db
            if i > 0:
                dz = dz_ant

    #Recibe una frase y devuelve la probabilidad de cada emocion
    def predecir(self, X):
        return self.fd(X)

"""Separa los datos en lotes de 256 frases.   
Con decenas de miles de frases, la matriz completa seria enorme
y podria llenar la memoria de la computadora por eso se separa en lotes"""
def obtener_probabilidades(red, textos, etiquetas, idx, n_clases):
    P = []#Guarda las probabilidades predicas de cada lote
    Y = []#Guarda las respuetas reales de cada lote
    #Recorre las frases por grupos pequeños para no llenar la memoria  
    for inicio in range(0, len(textos), BATCH):
        fin = inicio + BATCH#Marca el final del grupo actual
        X, y = crear_lote(textos[inicio:fin], etiquetas[inicio:fin], idx, n_clases) 
        P.append(red.fd(X))#Agrega las prediccion del grupo actual a P
        Y.append(y)#Guarda las respuestas al lote Y
    return np.vstack(P), np.vstack(Y)#Une todos los grupos en una sola tabla 

"""Busca el mejor umbral (nivel minimo de seguridad) para cada emocion por separado.
Asi las emociones raras pueden tener un umbral mas bajo y ser detectadas."""
def buscar_umbrales(P, Y):
    n_clases = Y.shape[1]#Cuenta cuantas emociones hay 
    umbrales = np.full(n_clases, UMBRAL)#Si algo falla se queda en 0.5
    for c in range(n_clases):#Revisa cada emocion una por una
        real = Y[:, c] >= 0.5#~Saca en qeu grases estaba la emocion
        mejor = -1
        #Prueba varios umbrales desde 0.05 hasta 0.05
        for u in np.arange(0.05, 0.96, 0.05):
            pred = P[:, c] >= u#Si la red supero el umbral la emocion esta
            tp = np.sum(pred & real)#Cuenta los aciertos
            fp = np.sum(pred & ~real)#Cuenta las falsas alarmas
            fn = np.sum(~pred & real)#Cuenta las que se les escaparon
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            if f1 > mejor:#Guardamos el umbral que mejor nota saco
                mejor = f1
                umbrales[c] = u
    return umbrales#Devuleve los umbrales ideales apra cada emocion


#Evalua a la red con frases que no vio en el entrenamiento
def evaluar(red, textos, etiquetas, idx, n_clases, pesos, umbrales):
    total_loss = 0
    total = 0
    #Guarda los acieros, las falsas alarmas, y las emociones que se escaparon
    tp = np.zeros(n_clases)
    fp = np.zeros(n_clases)
    fn = np.zeros(n_clases)
    for inicio in range(0, len(textos), BATCH):
        fin = inicio + BATCH
        X, Y = crear_lote(textos[inicio:fin], etiquetas[inicio:fin], idx, n_clases)
        P = red.fd(X)#Pide ala pred que predija las probabilidades
        total_loss += perdida(red.Z[-1], Y, pesos) * len(Y)
        total += len(Y)
        pred = P >= umbrales#Cada emocion se compara con SU umbral
        real = Y >= 0.5
        #Suma los acieros, falsas alarmas, las que se les escaparon por las emocines
        tp += np.sum(pred & real, axis=0)
        fp += np.sum(pred & ~real, axis=0)
        fn += np.sum(~pred & real, axis=0)
    loss = total_loss / total#Calcula el error promedio por frases
    macro, micro, f1_clases = calcular_f1(tp, fp, fn)
    return loss, macro, micro, f1_clases#Devuelve el error promedio de las 3

#Es el bucle principal, donde entrena y se corrige en varias vueltas
def entrenar(red, d, epocas=EPOCAS):
    #Guardamos los resultados para graficar loss y F1
    h = {"loss_tr": [], "loss_va": [], "f1_tr": [], "f1_va": []}
    n = len(d["Xtr"])
    n_clases = len(d["cats"])
    pesos = d["pesos"]#Importancia de cada emocion
    umbrales = np.full(n_clases, UMBRAL)#Durante el entrenamiento usamos 0.5 para todas
    for ep in range(epocas):
        orden = np.random.permutation(n)#Mezclamos las frases en cada vuelta
        total_loss = 0#Suma el error de esta vuelta
        total = 0
        #Guarda los aciertos, falsas alarmas y emociones que no vio
        tp = np.zeros(n_clases)
        fp = np.zeros(n_clases)
        fn = np.zeros(n_clases)
        for inicio in range(0, n, BATCH):
            posiciones = orden[inicio:inicio + BATCH]
            textos = [d["Xtr"][i] for i in posiciones]
            etiquetas = [d["Ytr"][i] for i in posiciones]
            X, Y = crear_lote(textos, etiquetas, d["idx"], n_clases)
            P = red.fd(X)#La red responde
            total_loss += perdida(red.Z[-1], Y, pesos) * len(Y)#Medimos que tan grande fue el error
            total += len(Y)#Suma cuantas frases iban en este grupo
            pred = P >= umbrales
            real = Y >= 0.5
            #Suma los acieros, falsas alarmas y las emociones que se les escaparon
            tp += np.sum(pred & real, axis=0)
            fp += np.sum(pred & ~real, axis=0)
            fn += np.sum(~pred & real, axis=0)
            red.bk(Y, pesos)#La red corrige los pesos y bias
        loss_tr = total_loss / total
        f1_tr, _, _ = calcular_f1(tp, fp, fn)
        #Evalua la red con datos nuevos
        loss_va, f1_va, micro_va, _ = evaluar(red, d["Xva"], d["Yva"], d["idx"], n_clases, pesos, umbrales)
        h["loss_tr"].append(loss_tr)
        h["loss_va"].append(loss_va)
        h["f1_tr"].append(f1_tr)
        h["f1_va"].append(f1_va)
        print(f"época {ep:3d} | loss tr {loss_tr:.4f} va {loss_va:.4f} | F1 tr {f1_tr:.4f} va {f1_va:.4f} | F1 micro va {micro_va:.4f}")
    return h#Devolvemos el historial completo para poder graficar


"""Creamos una funcion para guardar los datos y metricas del entrenamiento 
para que cada vez que ejecutamos el codijo no se vuelva a entrenar de nuevo"""
def guardar_modelo(red, d, umbrales):
    pesos = {}
    for i in range(len(red.W)):
        pesos[f"W{i}"] = red.W[i]
        pesos[f"b{i}"] = red.b[i]
    np.savez("goemotions_pesos.npz", **pesos)#Pesos y bias de la red
    with open("goemotions_vocab.json", "w", encoding="utf-8") as archivo:
        json.dump(d["idx"], archivo, ensure_ascii=False)#Diccionario de palabras
    with open("goemotions_emociones.json", "w", encoding="utf-8") as archivo:
        json.dump(d["cats"], archivo, ensure_ascii=False)#Nombres de las emociones
    with open("goemotions_umbrales.json", "w", encoding="utf-8") as archivo:
        json.dump([float(u) for u in umbrales], archivo)#Umbral de cada emocion
    print("\nModelo guardado correctamente.")

#Funcion para cargar el modelo que ya fue entrenado
def cargar_modelo():
    #Recuperamos los pesos y bias de cada capa
    with np.load("goemotions_pesos.npz") as archivo:
        W = [archivo[f"W{i}"].copy() for i in range(len(OCULTAS) + 1)]
        b = [archivo[f"b{i}"].copy() for i in range(len(OCULTAS) + 1)]
    #Recuperamos el diccionario, las emociones y sus umbrales
    with open("goemotions_vocab.json", encoding="utf-8") as archivo:
        idx = json.load(archivo)
    with open("goemotions_emociones.json", encoding="utf-8") as archivo:
        cats = json.load(archivo)
    with open("goemotions_umbrales.json", encoding="utf-8") as archivo:
        umbrales = np.array(json.load(archivo), dtype=np.float32)
    #Reconstruimos el vocabulario en sus pociciones originales
    vocab = [p for p, i in sorted(idx.items(), key=lambda x: x[1])]
    #Creamos la red y recuperamos sus parametros sin inicializar pesos nuevos
    red = RedNeuronal.__new__(RedNeuronal)
    red.lr = LR
    red.W = W
    red.b = b
    d = {"vocab": vocab, "idx": idx, "cats": cats}
    print("\nModelo cargado correctamente.")
    return red, d, umbrales

"""Prueba la red con una frase nueva. Detecta una emocion si supera su propio umbral
y las ordena segun que tanto superan ese limite,
evitando que emociones muy comunes como 'neutral' ganen siempre"""
def probar_frase(frase, red, d, umbrales):
    tokens = [limpiar(frase)]#Limpia la grase y la separa en palabras 
    X, _ = crear_lote(tokens, [[]], d["idx"], len(d["cats"]))#Convertimos la frase en ceros y unos
    probabilidades = red.predecir(X)[0]#Probabilidad de cada emocion
    relativo = probabilidades / umbrales#Que tan arriba de su umbral esta cada una
    orden = np.argsort(relativo)[::-1]#Ordena las emociones de mayor a menor
    print(f"\nFrase: {frase}")
    print("Emociones detectadas:")
    detectadas = 0
    for i in orden:
        if probabilidades[i] >= umbrales[i]:
            print(f"{d['cats'][i]:20s} {probabilidades[i] * 100:.2f}% (umbral {umbrales[i] * 100:.0f}%)")
            detectadas += 1
    if detectadas == 0:
        print("Ninguna emoción supero su umbral")
    print("\nLas 3 emociones mas probables.")
    for i in orden[:3]:
        print(f"{d['cats'][i]:20s} {probabilidades[i] * 100:.2f}% (umbral {umbrales[i] * 100:.0f}%)")


#Aqui comienza a ejecutarse todo el programa
if __name__ == "__main__":
    if MODO == "entrenar":#Separa los datos, entrena la red, guarda el modelo
        np.random.seed(SEMILLA)#Para que siempre salgan los mismos resultados
        datos, emociones = cargar_datos()#LLamamos a las frases en la DB
        print("Frases:", len(datos))
        print("Emociones:", len(emociones))
        #Limpiamos y dividimos los datos
        d = preparar_datos(datos, emociones)
        print("\nPalabras:", len(d["vocab"]))
        print("Train:", len(d["Xtr"]))
        print("Validación:", len(d["Xva"]))
        print("Test:", len(d["Xte"]))
        #Diseñamos la arquitectura de la red
        capas = [len(d["vocab"])] + OCULTAS + [len(d["cats"])]
        print("\nArquitectura:", capas)
        red = RedNeuronal(capas, lr=LR)#Llamamos al objeto RedNeuronal
        #Comienza el entrenamiento
        h = entrenar(red, d, EPOCAS)

        #Graficas del entrenamiento
        plt.figure(figsize=(10, 5))
        plt.plot(h["loss_tr"], label="loss entrenamiento")
        plt.plot(h["loss_va"], label="loss validación")
        plt.xlabel("época")
        plt.ylabel("loss")
        plt.title("Entrenamiento GoEmotions")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.show()

        #Buscamos el mejor umbral de cada emocion usando validacion
        n_clases = len(d["cats"])
        Pva, Yva = obtener_probabilidades(red, d["Xva"], d["Yva"], d["idx"], n_clases)
        umbrales = buscar_umbrales(Pva, Yva)
        print("\nUmbral encontrado por emocion")
        print("-" * 50)
        for nombre, u in zip(d["cats"], umbrales):
            print(f"{nombre:20s} umbral = {u:.2f}")

        #Evaluacion final, comparamos umbral fijo contra umbral por emocion
        print("\nEvaluacion de umbrales")
        print("-" * 50)
        fijo = np.full(n_clases, UMBRAL)
        _, macro0, micro0, _ = evaluar(red, d["Xte"], d["Yte"], d["idx"], n_clases, d["pesos"], fijo)
        print(f"Con umbral fijo 0.5     -> F1 macro: {macro0:.4f} | F1 micro: {micro0:.4f}")
        loss, macro, micro, f1_clases = evaluar(red, d["Xte"], d["Yte"], d["idx"], n_clases, d["pesos"], umbrales)
        print(f"Con umbral por emoción  -> F1 macro: {macro:.4f} | F1 micro: {micro:.4f}")
        print("Loss:", round(loss, 4))
        print("\nF1 POR EMOCION (umbral por emoción)")
        for nombre, valor in zip(d["cats"], f1_clases):
            print(f"{nombre:20s} F1 = {valor:.4f}")
        guardar_modelo(red, d, umbrales)#Guardamos el modelo

    elif MODO == "predecir":#Carga el modelo sin volver a entrenarlo
        red, d, umbrales = cargar_modelo()
    else:
        raise ValueError("MODO debe ser 'entrenar' o 'predecir'.")

    #Ponemos a prueba a la red con frases nuevas
    frases_prueba = [
        "The truth is I dont want to lose you",
        "You are very important in my life",
        "Yesterday you did recognize my voice",
        "I really do value your support whats going on with you",
        "I need to win you over"
    ]
    print("\nPonemos a pruba la red con frases reales")
    print("=" * 50)
    for frase in frases_prueba:
        probar_frase(frase, red, d, umbrales)

#Como se veria la red
def dibujar_red(d, ver_entrada=200):
    fig, ax = plt.subplots(figsize=(24, 14))
    n_entrada = len(d["vocab"])
    n_oculta1 = OCULTAS[0]
    n_oculta2 = OCULTAS[1]
    n_salida = len(d["cats"])
    ver_entrada = min(ver_entrada, n_entrada)#De la entrada dibujamos solo una parte
    #Posicion de cada nodo (x = la capa, y = la altura), todas las capas usan la misma altura total
    input_nodes = [(1, y) for y in np.linspace(0, 10, ver_entrada)]
    hidden1_nodes = [(5, y) for y in np.linspace(0, 10, n_oculta1)]
    hidden2_nodes = [(9, y) for y in np.linspace(0, 10, n_oculta2)]
    output_nodes = [(13, y) for y in np.linspace(0, 10, n_salida)]
    #Guardamos las capas en orden para unirlas de dos en dos
    capas_nodos = [input_nodes, hidden1_nodes, hidden2_nodes, output_nodes]
    lineas = []
    for g in range(len(capas_nodos) - 1):
        for a in capas_nodos[g]:
            for b in capas_nodos[g + 1]:
                lineas.append([a, b])#Cada linea es un par de puntos: donde empieza y donde termina
    ax.add_collection(LineCollection(lineas, colors="gray", linewidths=0.3, alpha=0.08, zorder=1))#Todas las conexiones de una vez
    #Dibujamos los nodos de cada capa, mas chicos mientras mas nodos hay
    ax.scatter(*zip(*input_nodes), s=25, color="skyblue", edgecolors="black", linewidths=0.4, zorder=2)
    ax.scatter(*zip(*hidden1_nodes), s=60, color="lightgreen", edgecolors="black", linewidths=0.5, zorder=2)
    ax.scatter(*zip(*hidden2_nodes), s=90, color="khaki", edgecolors="black", linewidths=0.6, zorder=2)
    ax.scatter(*zip(*output_nodes), s=140, color="salmon", edgecolors="black", linewidths=0.8, zorder=2)
    #Titulos de cada capa, indicando cuantos nodos reales tiene
    ax.text(1, 10.8, f"Entrada\n({n_entrada} palabras)\nse dibujan {ver_entrada}", ha="center", fontsize=12, fontweight="bold")
    ax.text(5, 10.8, f"Oculta 1 (ReLU)\n({n_oculta1} neuronas)", ha="center", fontsize=12, fontweight="bold")
    ax.text(9, 10.8, f"Oculta 2 (ReLU)\n({n_oculta2} neuronas)", ha="center", fontsize=12, fontweight="bold")
    ax.text(13, 10.8, f"Salida (Sigmoid)\n({n_salida} emociones)", ha="center", fontsize=12, fontweight="bold")
    #Nombre de cada emocion al lado de su nodo de salida
    for i, nombre in enumerate(d["cats"]):
        ax.text(13.3, output_nodes[i][1], nombre, ha="left", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(-0.7, 12.3)
    ax.axis("off")
    plt.title(f"Red Neuronal Clasificadora de Emociones ({n_entrada} → {n_oculta1} → {n_oculta2} → {n_salida})", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("goemotions_red.png", dpi=200)#Guardamos la imagen para el video
    plt.show()

dibujar_red(d, ver_entrada=100)