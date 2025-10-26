import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.utils.run_manager import RunManager

def load_scin_data(scin_root: Path) -> pd.DataFrame:
    """Loads and unnests the SCIN dataset metadata.

    Args:
        scin_root: Path to scin directory (e.g., data/scin)
    """
    print("Loading SCIN data...")
    scin_df = pd.read_csv(scin_root / "dataset" / "scin_cases.csv")

    # Only melt columns that contain actual image paths (image_X_path)
    path_cols = [col for col in scin_df.columns if col.startswith('image_') and col.endswith('_path')]
    id_vars = [col for col in scin_df.columns if col not in path_cols]

    long_df = pd.melt(scin_df, id_vars=id_vars, value_vars=path_cols,
                      value_name='image_path', var_name='image_num')
    long_df = long_df.dropna(subset=['image_path'])
    # Paths in CSV are relative to scin_root (e.g., "dataset/images/xxx.png")
    long_df['image_path'] = long_df.apply(lambda row: scin_root / row['image_path'], axis=1)
    long_df['dataset'] = 'scin'

    return long_df

def load_generic_data(path: Path, dataset_name: str) -> pd.DataFrame:
    """Loads a dataset with a simple directory structure."""
    print(f"Loading {dataset_name} data...")
    image_paths = list(path.glob("*.jpg"))
    df = pd.DataFrame({'image_path': image_paths})
    df['dataset'] = dataset_name
    return df

def main(data_dir: Path, output_dir: Path, limit: int = None, force: bool = False):
    """Main function to load all datasets and save a master list.

    Args:
        data_dir: Directory containing the datasets
        output_dir: Directory to save the output CSV
        limit: Maximum number of images to load (for testing). If None, loads all images.
        force: Recompute even if cached output exists
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manager = RunManager(output_dir)
    output_name = "master_image_list.csv"
    output_path = output_dir / output_name

    if not force and output_path.exists() and limit is None:
        print(f"Cached master list found at {output_path}. Use --force to regenerate.")
        return

    if force:
        archived = run_manager.archive_previous_outputs([output_name])
        if archived:
            print(f"Archived previous master list to {run_manager.old_dir}")

    scin_df = load_scin_data(data_dir / "scin")
    isic_df = load_generic_data(data_dir / "isic", "isic")
    ddi_df = load_generic_data(data_dir / "ddi", "ddi")

    master_df = pd.concat([scin_df, isic_df, ddi_df], ignore_index=True)

    # Apply limit if specified (for testing/development)
    if limit is not None:
        original_count = len(master_df)
        master_df = master_df.sample(n=min(limit, len(master_df)), random_state=42)
        print(f"Limited dataset from {original_count} to {len(master_df)} images for testing")

    master_df.to_csv(output_path, index=False)
    print(f"Master image list saved to {output_path}")
    print(f"Total images loaded: {len(master_df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and consolidate dermatology datasets.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Directory containing the datasets.")
    parser.add_argument("--output_dir", type=Path, default=Path("reports"), help="Directory to save the output CSV.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images for testing (default: load all)")
    parser.add_argument("--force", action="store_true", help="Recompute master list even if cached output exists.")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.limit, force=args.force)
