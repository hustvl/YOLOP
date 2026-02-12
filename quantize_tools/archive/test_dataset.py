import os
import random
from pathlib import Path

# --- CONFIGURATION DES CHEMINS ---
BASE_DIR = os.getcwd()
IMG_DIR = os.path.join(BASE_DIR, "dataset", "images", "val")
DA_DIR  = os.path.join(BASE_DIR, "dataset", "da_seg_annotations", "val")
LL_DIR  = os.path.join(BASE_DIR, "dataset", "ll_seg_annotations", "val")
DET_DIR = os.path.join(BASE_DIR, "dataset", "det_annotations", "val")

NB_SAMPLES = 200

def check_concordance():
    if not os.path.exists(IMG_DIR):
        print(f"❌ Erreur : Le dossier d'images n'existe pas : {IMG_DIR}")
        return

    # Liste toutes les images .jpg
    all_images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith('.jpg')]
    
    if len(all_images) == 0:
        print(f"❌ Erreur : Aucune image .jpg trouvée dans {IMG_DIR}")
        return

    print(f"🔍 Analyse de {min(NB_SAMPLES, len(all_images))} images sur un total de {len(all_images)}...")

    # Tirage aléatoire
    samples = random.sample(all_images, min(NB_SAMPLES, len(all_images)))

    missing_da = 0
    missing_ll = 0
    missing_det = 0
    errors = []

    for img_name in samples:
        # Nom de base sans extension (ex: 7dd9ef45-f197db95)
        base_name = os.path.splitext(img_name)[0]
        
        # Chemins attendus (YOLOP utilise généralement .png pour les masques)
        da_path  = os.path.join(DA_DIR, base_name + ".png")
        ll_path  = os.path.join(LL_DIR, base_name + ".png")
        # Pour les détections, cela peut être .json ou .txt selon ta version
        det_path_json = os.path.join(DET_DIR, base_name + ".json")
        det_path_txt  = os.path.join(DET_DIR, base_name + ".txt")

        # Vérifications
        da_exists = os.path.exists(da_path)
        ll_exists = os.path.exists(ll_path)
        det_exists = os.path.exists(det_path_json) or os.path.exists(det_path_txt)

        if not da_exists:
            missing_da += 1
        if not ll_exists:
            missing_ll += 1
        if not det_exists:
            missing_det += 1

        if not (da_exists and ll_exists and det_exists):
            errors.append({
                "img": img_name,
                "da": "OK" if da_exists else "MANQUANT",
                "ll": "OK" if ll_exists else "MANQUANT",
                "det": "OK" if det_exists else "MANQUANT"
            })

    # --- RAPPORT FINAL ---
    print("\n" + "="*50)
    print("📊 RAPPORT DE CONCORDANCE DU DATASET")
    print("="*50)
    print(f"Images vérifiées      : {len(samples)}")
    print(f"Masques Route (DA) manquants : {missing_da}")
    print(f"Masques Lignes (LL) manquants: {missing_ll}")
    print(f"Labels Détection manquants    : {missing_det}")
    print("-" * 50)

    if len(errors) > 0:
        print("⚠️ Détails des premières erreurs trouvées :")
        for err in errors[:10]: # Affiche les 10 premières erreurs
            print(f"  - {err['img']} -> DA: {err['da']} | LL: {err['ll']} | DET: {err['det']}")
        if len(errors) > 10:
            print(f"  ... et {len(errors) - 10} autres erreurs.")
    else:
        print("✅ Parfait ! Toutes les images ont leurs fichiers correspondants.")
    print("="*50)

if __name__ == "__main__":
    check_concordance()