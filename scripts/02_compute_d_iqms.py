"""
This script computes the Dermatologic Image Quality Metrics (D-IQMs)
for all images in the master list and saves them to a feature matrix.
"""

import pandas as pd
from pathlib import Path
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from src.diqa.metrics import (
    compute_sharpness_laplacian,
    compute_exposure_histogram,
    compute_glare_hsv,
)

def process_image(image_path_str: str) -> dict:
    """
    Worker function to compute all D-IQMs for a single image.
    Takes a string path and converts it to a Path object.
    """
    image_path = Path(image_path_str)
    try:
        sharpness = compute_sharpness_laplacian(image_path)
        under_exposed, over_exposed = compute_exposure_histogram(image_path)
        glare = compute_glare_hsv(image_path)
        
        return {
            'image_path': image_path_str,
            'sharpness': sharpness,
            'under_exposed': under_exposed,
            'over_exposed': over_exposed,
            'glare': glare,
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return {
            'image_path': image_path_str,
            'sharpness': None,
            'under_exposed': None,
            'over_exposed': None,
            'glare': None,
        }

def main(master_list_path: Path, output_dir: Path):
    """
    Main function to process all images in parallel and save the feature matrix.
    """
    df = pd.read_csv(master_list_path)
    image_paths = df['image_path'].tolist()

    print(f"Starting D-IQM computation for {len(image_paths)} images...")
    
    # Use all available CPU cores
    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(process_image, image_paths), total=len(image_paths)))

    results_df = pd.DataFrame(results)
    
    # Merge with original data to keep metadata
    merged_df = pd.merge(df, results_df, on='image_path', how='left')
    
    output_path = output_dir / "feature_matrix.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"Feature matrix saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute D-IQMs for a list of images.")
    parser.add_argument("--master_list_path", type=Path, default=Path("reports/master_image_list.csv"), help="Path to the master image list CSV.")
    parser.add_argument("--output_dir", type=Path, default=Path("reports"), help="Directory to save the output feature matrix.")
    args = parser.parse_args()
    main(args.master_list_path, args.output_dir)
