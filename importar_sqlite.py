import sqlite3
import csv
from pathlib import Path

carpeta = Path("goemotions_data")
conexion = sqlite3.connect("goemotions.db")
cursor = conexion.cursor()
cursor.execute("PRAGMA foreign_keys = ON")

#Crear tablas
cursor.executescript("""
CREATE TABLE IF NOT EXISTS muestras (
    id TEXT PRIMARY KEY,
    texto TEXT NOT NULL,
    conjunto TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS emociones (
    id INTEGER PRIMARY KEY,
    nombre TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS muestra_emocion (
    muestra_id TEXT,
    emocion_id INTEGER,
    PRIMARY KEY (muestra_id, emocion_id),
    FOREIGN KEY (muestra_id) REFERENCES muestras(id),
    FOREIGN KEY (emocion_id) REFERENCES emociones(id)
);
""")

#Importar las 28 emociones
with open(carpeta / "emotions.txt", encoding="utf-8") as archivo:
    emociones = [line.strip() for line in archivo if line.strip()]
    
for i, emocion in enumerate(emociones):
    cursor.execute(
        "INSERT OR IGNORE INTO emociones VALUES (?, ?)",
        (i, emocion)
    )

#Importar los tres conjuntos
for conjunto in ["train", "dev", "test"]:
    ruta = carpeta / f"{conjunto}.tsv"
    with open(ruta, encoding="utf-8", newline="") as archivo:
        lector = csv.reader(archivo, delimiter="\t")

        for fila in lector:
            texto, etiquetas, identificador = fila
            cursor.execute("""
                INSERT OR IGNORE INTO muestras
                VALUES (?, ?, ?)
            """, (identificador, texto, conjunto))

            for etiqueta in etiquetas.split(","):
                cursor.execute("""
                    INSERT OR IGNORE INTO muestra_emocion
                    VALUES (?, ?)
                """, (identificador, int(etiqueta)))

    print(f"Importado: {conjunto}")

conexion.commit()

#Mostrar resultados
for tabla in ["muestras", "emociones", "muestra_emocion"]:
    cursor.execute(f"SELECT COUNT(*) FROM {tabla}")
    print(f"{tabla}: {cursor.fetchone()[0]:,}")

conexion.close()

print("\nBase de datos creada correctamente.")