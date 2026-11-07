from pyngrok import ngrok
import time

# 1. Autenticación con tu token de ngrok
# ngrok.set_auth_token("3LlDKm7lShwySVzztBq6eLBQiv4_3pshys3KcFYTDhtTs4ysu")

# 2. Crear el túnel apuntando al puerto de Streamlit (8501)
public_url = ngrok.connect(8501)
print("\n" + "="*50)
print(f"TU ENLACE PÚBLICO ES: {public_url.public_url}")
print("="*50 + "\n")
print("Mantén esta ventana abierta. Presiona Ctrl+C para cerrar el túnel.")

# 3. Bucle para mantener el túnel abierto
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Cerrando el túnel...")
    ngrok.kill()