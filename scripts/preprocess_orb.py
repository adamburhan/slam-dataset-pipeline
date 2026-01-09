import cv2
import numpy as np
import argparse
from pathlib import Path
from sklearn.decomposition import PCA
import joblib
import os

# --- Configuration ---
N_FEATURES = 2000 
SCALE_FACTOR = 1.2
N_LEVELS = 8
PCA_COMPONENTS = 3
PCA_SAMPLE_SIZE = 5000  # Number of random images to learn the "Global Color Palette"

def get_orb():
    return cv2.ORB_create(
        nfeatures=N_FEATURES,
        scaleFactor=SCALE_FACTOR,
        nlevels=N_LEVELS,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_FAST_SCORE,
        patchSize=31,
        fastThreshold=20
    )

def train_pca_global(dataset_root):
    """Scans the ENTIRE dataset to train a robust, global PCA."""
    print(f"--- Training Global PCA on {dataset_root} ---")
    print("Scanning for images (this might take a moment)...")
    
    # 1. Gather ALL images from ALL sequences
    root_path = Path(dataset_root)
    all_images = sorted(list(root_path.rglob("*.png")))
    if not all_images:
        all_images = sorted(list(root_path.rglob("*.jpg")))
        
    if not all_images:
        raise ValueError(f"No images found in {dataset_root}")

    # 2. Randomly Sample 5,000 images
    # This ensures we see rain, sun, tunnels, and highways
    sample_size = min(len(all_images), PCA_SAMPLE_SIZE)
    print(f"Found {len(all_images)} total images. Sampling {sample_size} for training...")
    
    train_files = np.random.choice(all_images, sample_size, replace=False)
    
    # 3. Compute Descriptors
    orb = get_orb()
    descriptors = []
    
    for i, p in enumerate(train_files):
        if i % 100 == 0: print(f"Processing sample {i}/{sample_size}...", end="\r")
        
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        _, des = orb.detectAndCompute(img, None)
        if des is not None:
            descriptors.append(des)
            
    print("\nStacking descriptors...")
    if not descriptors:
        return None

    X = np.vstack(descriptors).astype(np.float32)
    
    # 4. Fit PCA
    print(f"Fitting PCA on {X.shape[0]} descriptors...")
    pca = PCA(n_components=PCA_COMPONENTS)
    pca.fit(X)
    
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    return pca

def generate_semantic_map(image_shape, keypoints, descriptors, pca):
    """Generates the colored dot map."""
    feature_map = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    if not keypoints or descriptors is None: return feature_map

    des_reduced = pca.transform(descriptors.astype(np.float32))
    des_normalized = ((np.clip(des_reduced, -100, 100) + 100) / 200.0 * 255).astype(np.uint8)

    for i, kp in enumerate(keypoints):
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            color = des_normalized[i].tolist() 
            cv2.circle(feature_map, (x, y), 3, color, -1) 
    return feature_map

def process_sequence(seq_dir, output_root, pca):
    seq_path = Path(seq_dir)
    
    # Handle naming (KITTI vs TartanAir)
    if seq_path.name in ["image_0", "image_left", "rgb"]:
        seq_name = seq_path.parent.name
    else:
        seq_name = seq_path.name
    
    output_dir = Path(output_root) / seq_name / "orb_semantic"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img_files = sorted(list(seq_path.glob("*.png")))
    if not img_files: img_files = sorted(list(seq_path.glob("*.jpg")))
    
    print(f"Processing {seq_name} ({len(img_files)} frames)...")
    orb = get_orb()
    
    for img_path in img_files:
        save_path = output_dir / img_path.name
        if save_path.exists(): continue
        
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        kp, des = orb.detectAndCompute(img, None)
        sem_map = generate_semantic_map(img.shape, kp, des, pca)
        cv2.imwrite(str(save_path), sem_map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--pca_path", default="orb_pca.joblib")
    args = parser.parse_args()
    
    root = Path(args.dataset_root)
    
    # --- 1. Global PCA Handling ---
    if os.path.exists(args.pca_path):
        print(f"Loading Global PCA from {args.pca_path}...")
        pca = joblib.load(args.pca_path)
    else:
        # If missing, we train on the WHOLE dataset provided
        pca = train_pca_global(root)
        if pca is not None:
            joblib.dump(pca, args.pca_path)
            print(f"Global PCA saved to {args.pca_path}")
        else:
            print("Error: Could not train PCA.")
            exit()

    # --- 2. Process All Sequences ---
    seq_dirs = (
        sorted(list(root.rglob("image_0"))) +   # KITTI
        sorted(list(root.rglob("image_left"))) + # TartanAir
        sorted(list(root.rglob("data"))) +      # EuRoC
        sorted(list(root.rglob("rgb")))         # ETH3D
    )

    for seq in seq_dirs:
        if seq.exists():
            process_sequence(seq, args.output_root, pca)
        
    print("Done.")