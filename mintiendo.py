import numpy as np
import matplotlib.pyplot as plt

"""Red neuroanl para saber si te estan mintiendo por CHAT"""
'Ren con arquitectura de 6,8,8,4,1 o 6 entradas 3 capas ocultas y una salida'

#Tiempo que tarda en responder 
#Cambios en la historía que cuenta
#Contradicciones en lo que dice
#Respuestas evasivas
#Exceso de dealles innecesarios 
#Cambios en su forma habituergdggasdal de escribir 


#Funciones de activacion 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0,z) 

#Funcion que ayuda a saber hacia donde corregir los pesos 
def relu_deriv(z):
    return (z > 0).astype(float)

#Calcula que tanto se equivoco la red
def bce(y_real, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1-1e7)
    return -np.mean(y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred))

np.random.seed(42)

'''Ahora voy a crear una clase de red neuronal ya que crear asi las redes neuronales 
es el estandar es mas limpio y mas facil de mantener el codijo, importante saber 
programacion orientada a objetos'''

#Red neuronal
class RedNeuronal:
    def __init__(self):
        self.capas = []#Aqui se guardan todas las capas
        #Agrega una nueva capa ala red
        def add_layer(self, n_in, n_out,activacion="relu"):
            #Inicializamos los pesos con He
            if activacion == "relu":
                w = np.random.randn(n_out, n_in) * np.sqrt(2 / n_in)
            else:
                w = np.random.randn(n_out, n_in) * np.sqrt(1 / n_in)
                
            b = np.zeros(n_out)#El bias comienza en cero
            #Guarda la informacion de la capa
            self.capas.apped({"W":w, "b":b, "activacion":activacion})
        
        #PROPAGACIÓN HACIA ADELANTE 
        def fd(self,x):
            #Aqui se guarda los datos para el aprendizaje después
            self.cache = []
            a = x#La entrda comienza siendo x
            for capa in self.capas:
                a_in = a#Guarda la entrada de esta capa
                #Calcula la combinacion de pesos y entrada
                z = capa["w"] @ a + capa["b"]
                #Aplica la función de activación
                if capa["activacion"] == "relu":
                    a = relu(z)
                else:
                    a = sigmoid(z)
                #Guarda los resultados para utlizarlos durante el aprendizaje
                self.cache.append({"z":z, "a":a, "a_in":a_in})
            #Devuelve la prediccion de la re 
            return a 
        #PROPAGACIÓN HACIA ATRAS 
        def bk(self, y, alpha):
            #Aqui se guardan los gradientes
            grads = []
            a_out = self.cache[-1]["a"]#Comenzamos desde la ultima capa
            delta = a_out - y#Calcula el error de salida
            #Recorre las capas de atrás hacia adelante
            for i in reversed(range(len(self.capas))):
                capa = self.capas[i]
                z = self.cache[i]["a_in"]
                a_in = self.cache[i]["a_in"]
                #Calcula cuanto debe cambiar cada peso 
                dw = np.outer(delta,a_in)
                #Calcula cuanto debe cambiar cada bias
import numpy as np 
import matplotlib.pyplot as plt 
 
"""Red neuroanl para saber si te estan mintiendo por CHAT""" 
'Ren con arquitectura de 6,8,8,4,1 o 6 entradas 3 capas ocultas y una salida' 
 
#Tiempo que tarda en responder  
#Cambios en la historía que cuenta 
#Contradicciones en lo que dice 
#Respuestas evasivas 
#Exceso de dealles innecesarios  
#Cambios en su forma habituergdggasdal de escribir  
 
 
#Funciones de activacion  
def sigmoid(z): 
    return 1 / (1 + np.exp(-z)) 
 
def relu(z): 
    return np.maximum(0,z) 
 
#Funcion que ayuda a saber hacia donde corregir los pesos  
def relu_deriv(z): 
    return (z > 0).astype(float) 
 
#Calcula que tanto se equivoco la red 
def bce(y_real, y_pred): 
    y_pred = np.clip(y_pred, 1e-7, 1-1e-7) 
    return -np.mean(y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred)) 
 
np.random.seed(42) 
 
'''Ahora voy a crear una clase de red neuronal ya que crear asi las redes neuronales  
es el estandar es mas limpio y mas facil de mantener el codijo, importante saber  
programacion orientada a objetos''' 
 
#Red neuronal 
class RedNeuronal: 
    def __init__(self): 
        self.capas = []#Aqui se guardan todas las capas 
    #Agrega una nueva capa ala red 
    def add_layer(self, n_in, n_out,activacion="relu"): 
        #Inicializamos los pesos con He 
        if activacion == "relu": 
            w = np.random.randn(n_out, n_in) * np.sqrt(2 / n_in) 
        else: 
            w = np.random.randn(n_out, n_in) * np.sqrt(1 / n_in) 
             
        b = np.zeros(n_out)#El bias comienza en cero 
        #Guarda la informacion de la capa 
        self.capas.append({"w":w, "b":b, "activacion":activacion}) 
         
    #PROPAGACIÓN HACIA ADELANTE  
    def fd(self,x): 
        #Aqui se guarda los datos para el aprendizaje después 
        self.cache = [] 
        a = x#La entrda comienza siendo x 
        for capa in self.capas: 
            a_in = a#Guarda la entrada de esta capa 
            #Calcula la combinacion de pesos y entrada 
            z = capa["w"] @ a + capa["b"] 
            #Aplica la función de activación 
            if capa["activacion"] == "relu": 
                a = relu(z) 
            else: 
                a = sigmoid(z) 
            #Guarda los resultados para utlizarlos durante el aprendizaje 
            self.cache.append({"z":z, "a":a, "a_in":a_in}) 
        #Devuelve la prediccion de la re  
        return a  
        
    #PROPAGACIÓN HACIA ATRAS  
    def bk(self, y, alpha): 
        #Aqui se guardan los gradientes 
        grads = [] 
        a_out = self.cache[-1]["a"]#Comenzamos desde la ultima capa 
        delta = a_out - y#Calcula el error de salida 
        #Recorre las capas de atrás hacia adelante 
        for i in reversed(range(len(self.capas))): 
            capa = self.capas[i] 
            z = self.cache[i]["a_in"] 
            a_in = self.cache[i]["a_in"] 
            #Calcula cuanto debe cambiar cada peso  
            dw = np.outer(delta,a_in) 
            #Calcula cuanto debe cambiar cada bias 
            db = delta.copy() 
            #Guarda los gradientes 
            grads.insert(0,{"dw":dw, "db":db}) 
            #Si no estamos en la primera capa pasamos el error hacia atras 
            if i > 0: 
                da = capa["w"].T @ delta 
                #Recupera el z de la capa anterior 
                z_anterior = self.cache[i - 1]["z"] 
                delta = (da * relu_deriv(z_anterior)) 
        #ACTUALIZA LOS PESOS Y BIAS 
        for capa, g in zip(self.capas,grads): 
            capa["w"] -= (alpha * g["dw"]) 
            capa["b"] -= (alpha * g["db"]) 
                 
"""con esta arquitectura podemos crear capas sin tener que programar 
todas las funciones a mano""" 
 
rn = RedNeuronal() 
#Primera capa, 6 entradas 8 neuronas 
rn.add_layer(6,8, "relu") 
#Segunda capa, 8 neuronas 8 neuronas 
rn.add_layer(8,8, "relu") 
#Tercera Capa 8 neuronas 8 neuronas 
rn.add_layer(8,4, "relu") 
#Capa de salida 4 neuronas 1 neurona 
rn.add_layer(4,1, "sigmoid") 
npc = np.array([0.9, 0.7,0.9, 0.8, 0.9,0.9]) 
 
prediccion = rn.fd(npc) 
porcentaje = prediccion[0] * 100 
print(f"\nPuntuacion de señales: ",f"{porcentaje:.2f}%") 
if prediccion[0] >=0.5: 
    print("Lo mas posible que esta persona te mintio por chat") 
else: 
    print("Lo mas probable es que no te haya mentido por chat")
    
    
#Como se veria graficada 
fig, ax = plt.subplots(figsize=(10, 7))

input_nodes = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)]
hidden1_nodes = [(3, 0.5), (3, 1.25), (3, 2), (3, 2.75), (3, 3.5), (3, 4.25), (3, 5), (3, 5.75)]
hidden2_nodes = [(5, 0.5), (5, 1.25), (5, 2), (5, 2.75), (5, 3.5), (5, 4.25), (5, 5), (5, 5.75)]
hidden3_nodes = [(7, 2), (7, 2.9), (7, 3.8), (7, 4.7)]
output_node = (9, 3.35)

for i_node in input_nodes:
    for h_node in hidden1_nodes:
        ax.plot([i_node[0], h_node[0]], [i_node[1], h_node[1]], "gray", zorder=1)
for h1_node in hidden1_nodes:
    for h2_node in hidden2_nodes:
        ax.plot([h1_node[0], h2_node[0]], [h1_node[1], h2_node[1]], "gray", zorder=1)
for h2_node in hidden2_nodes:
    for h3_node in hidden3_nodes:
        ax.plot([h2_node[0], h3_node[0]], [h2_node[1], h3_node[1]], "gray", zorder=1)
for h3_node in hidden3_nodes:
    ax.plot([h3_node[0], output_node[0]], [h3_node[1], output_node[1]], "gray", zorder=1)
for node in input_nodes:
    ax.scatter(*node, s=800, color="skyblue", zorder=2)
for node in hidden1_nodes:
    ax.scatter(*node, s=700, color="lightgreen", zorder=2)
for node in hidden2_nodes:
    ax.scatter(*node, s=700, color="lightgreen", zorder=2)
for node in hidden3_nodes:
    ax.scatter(*node, s=800, color="orange", zorder=2)
ax.scatter(*output_node, s=900, color="salmon", zorder=2)
nombres = ["tiempo_respuesta", "cambios_historia", "contradicciones", "evasivas", "exceso_detalles", "cambios_forma_escribir"]

for i, nombre in enumerate(nombres):
    ax.text(0.8, input_nodes[i][1], nombre, ha="right", va="center", fontsize=9)

ax.text(1, 6.6, "Entrada\n6", ha="center", fontsize=11)
ax.text(3, 6.6, "Oculta 1\n8", ha="center", fontsize=11)
ax.text(5, 6.6, "Oculta 2\n8", ha="center", fontsize=11)
ax.text(7, 5.5, "Oculta 3\n4", ha="center", fontsize=11)
ax.text(9, 4.2, "Salida\n1", ha="center", fontsize=11)
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")
plt.title("Red neuronal 6 → 8 → 8 → 4 → 1")
plt.show()