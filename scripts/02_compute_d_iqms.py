"""
This script computes the Dermatologic Image Quality Metrics (D-IQMs)
for all images in the master list and saves them to a feature matrix.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from pathlib import Path
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

from src.diqa.metrics import (
    compute_sharpness_laplacian,
    compute_exposure_histogram,
    compute_glare_adaptive,
    compute_contrast,
    compute_color_metrics,
    compute_noise_level,
    compute_entropy,
    compute_edge_density,
    compute_dynamic_range,
    compute_brisque_features,
    compute_lesion_framing,
)
from scripts.utils.run_manager import RunManager

def process_image(image_path_str: str) -> dict:
    """
    Worker function to compute all D-IQMs for a single image.
    Takes a string path and converts it to a Path object.

    Computes 16 segmentation-free D-IQMs covering all dimensions
    identified in teledermatology IQA literature (2022-2025), plus
    BRISQUE features for state-of-art comparison.
    """
    image_path = Path(image_path_str)
    try:
        # Original 6 D-IQMs
        sharpness = compute_sharpness_laplacian(image_path)
        under_exposed, over_exposed = compute_exposure_histogram(image_path)
        glare = compute_glare_adaptive(image_path)
        lesion_size, lesion_centrality = compute_lesion_framing(image_path)

        # NEW: Critical derm-specific features (Tier 1)
        global_contrast, local_contrast = compute_contrast(image_path)
        color_variance, color_cast = compute_color_metrics(image_path)
        noise = compute_noise_level(image_path)
        entropy = compute_entropy(image_path)

        # NEW: Complementary features (Tier 2)
        edge_density = compute_edge_density(image_path)
        dynamic_range = compute_dynamic_range(image_path)

        # BRISQUE features (state-of-art comparison)
        brisque_var, brisque_skew = compute_brisque_features(image_path)

        return {
            'image_path': image_path_str,
            # Sharpness/Blur (2 features)
            'sharpness': sharpness,
            'edge_density': edge_density,
            # Exposure/Lighting (4 features)
            'under_exposed': under_exposed,
            'over_exposed': over_exposed,
            'dynamic_range': dynamic_range,
            'glare': glare,
            # Color (2 features - CRITICAL for derm, LAB space)
            'color_variance': color_variance,
            'color_cast': color_cast,
            # Contrast (2 features - CRITICAL for derm)
            'global_contrast': global_contrast,
            'local_contrast': local_contrast,
            # Noise & Information (4 features, includes BRISQUE)
            'noise': noise,
            'entropy': entropy,
            'brisque_variance': brisque_var,
            'brisque_skewness': brisque_skew,
            # Framing (2 features)
            'lesion_size': lesion_size,
            'lesion_centrality': lesion_centrality,
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return {
            'image_path': image_path_str,
            'sharpness': None,
            'edge_density': None,
            'under_exposed': None,
            'over_exposed': None,
            'dynamic_range': None,
            'glare': None,
            'color_variance': None,
            'color_cast': None,
            'global_contrast': None,
            'local_contrast': None,
            'noise': None,
            'entropy': None,
            'brisque_variance': None,
            'brisque_skewness': None,
            'lesion_size': None,
            'lesion_centrality': None,
        }

def main(master_list_path: Path, output_dir: Path, force: bool = False):
    """
    Main function to process all images in parallel and save the feature matrix.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manager = RunManager(output_dir)
    output_name = "feature_matrix.csv"
    output_path = output_dir / output_name

    need_recompute = force or run_manager.should_recompute(
        output_name,
        [master_list_path]
    )

    if not need_recompute:
        try:
            master_df = pd.read_csv(master_list_path)
            cached_df = pd.read_csv(output_path)
            master_unique = master_df['image_path'].astype(str).nunique()
            cached_unique = cached_df['image_path'].astype(str).nunique()
            if cached_unique >= master_unique:
                print(f"Cached feature matrix found at {output_path} (covers {cached_unique} images).")
                print("Skipping D-IQM computation. Use --force to recompute.")
                return
            else:
                print(f"Cached feature matrix covers {cached_unique}/{master_unique} images; recomputing missing features.")
                need_recompute = True
        except Exception as exc:
            print(f"Warning: unable to validate cached feature matrix ({exc}). Will recompute.")
            need_recompute = True

    if not need_recompute:
        return

    cached_output = run_manager.get_cached_output(output_name)
    if cached_output is not None and cached_output.exists():
        archived = run_manager.archive_previous_outputs([output_name])
        if archived:
            print(f"Archived previous feature matrix to {run_manager.old_dir}")

    start_time = time.time()

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

    elapsed_time = time.time() - start_time

    # Count successful extractions
    n_success = results_df['sharpness'].notna().sum()
    n_failed = len(results_df) - n_success

    # Log run
    run_manager.log_run(
        phase="02_compute_d_iqms",
        status="completed",
        inputs={"master_list": str(master_list_path), "n_images": len(image_paths)},
        outputs={"feature_matrix": str(output_path)},
        metadata={
            "n_success": int(n_success),
            "n_failed": int(n_failed),
            "n_cpus": cpu_count(),
            "processing_time_seconds": elapsed_time
        }
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute D-IQMs for a list of images.")
    parser.add_argument("--master_list_path", type=Path, default=Path("reports/master_image_list.csv"), help="Path to the master image list CSV.")
    parser.add_argument("--output_dir", type=Path, default=Path("reports"), help="Directory to save the output feature matrix.")
    parser.add_argument("--force", action="store_true", help="Recompute features even if cached output exists.")
    args = parser.parse_args()
    main(args.master_list_path, args.output_dir, force=args.force)
