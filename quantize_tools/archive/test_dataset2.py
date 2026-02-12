import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- CHEMINS CORRIGÉS (BASÉS SUR TES SCREENSHOTS) ---
# On pointe vers le dossier TRAIN et non VAL
IMG_PATH = "dataset/images/train/7dd9ef45-f197db95.jpg"
MASK_PATH = "dataset/da_seg_annotations/train/7dd9ef45-f197db95.png"

def check_visual():
    print(f"🔍 Recherche de l'image : {IMG_PATH}")
    
    if not os.path.exists(IMG_PATH):
        print("❌ Erreur : Image introuvable (Vérifie le dossier dataset/images/train)")
        return
    if not os.path.exists(MASK_PATH):
        print("❌ Erreur : Masque introuvable (Vérifie le dossier dataset/da_seg_annotations/train)")
        return

    print("✅ Fichiers trouvés ! Chargement...")

    # 1. Charger l'image et le masque
    img = Image.open(IMG_PATH).convert("RGB")
    mask = Image.open(MASK_PATH).convert("L") # Noir et Blanc
    mask_np = np.array(mask)

    print(f"📊 Stats du masque : Valeurs uniques = {np.unique(mask_np)}")

    # 2. Créer l'image de vérification
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Image Originale (Train)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.imshow(mask_np, alpha=0.5, cmap='jet') # Superposition
    plt.title(f"Masque Superposé (Valeurs: {np.unique(mask_np)})")
    plt.axis('off')

    # 3. Sauvegarder au lieu d'afficher (pour Docker)
    output_file = "verif_train.png"
    plt.savefig(output_file)
    print(f"🎉 Succès ! Ouvre l'image '{output_file}' pour voir le résultat.")

if __name__ == "__main__":
    check_visual()