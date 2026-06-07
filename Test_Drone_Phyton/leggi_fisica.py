import mujoco

model = mujoco.MjModel.from_xml_path("skydio_x2/x2.xml")

print("--- DATI FISICI ESTRATTI DA MUJOCO ---")
# La massa totale è la somma delle masse di tutti i body (telaio + rotori)
print(f"Massa totale (m_val): {sum(model.body_mass):.4f} kg")

# L'inerzia del corpo principale (il telaio del drone è il body 1)
inerzia = model.body_inertia[1]
print(f"Inerzia Ixx: {inerzia[0]:.6f}")
print(f"Inerzia Iyy: {inerzia[1]:.6f}")
print(f"Inerzia Izz: {inerzia[2]:.6f}")