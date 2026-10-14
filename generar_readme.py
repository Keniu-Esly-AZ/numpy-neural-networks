import os

base = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(base, "imagenes")
exts = (".jpeg", ".jpg", ".png")

scripts = sorted(
    f[:-3] for f in os.listdir(base)
    if f.endswith(".py") and f[:-3].lower() != "generar_readme"
)

lineas = [
    "# numpy-neural-networks",
    "",
    "Redes neuronales hechas con NumPy.",
    "",
    "## Proyectos",
    "",
]

for s in scripts:
    imagenes = sorted(
        f for f in os.listdir(img_dir)
        if f.startswith(s + "_") and f.lower().endswith(exts)
    )
    lineas += [f"### {s.replace('_', ' ')}", "", f"[Ver código]({s}.py)", ""]
    for img in imagenes:
        lineas += [f"![{s}](imagenes/{img})", ""]
    if not imagenes:
        print("Sin imagen:", s)

with open(os.path.join(base, "README.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lineas))

print("README.md generado con", len(scripts), "proyectos")