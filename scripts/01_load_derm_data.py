import pandas as pd
import glob
from pathlib import Path
import argparse

def load_scin_data(path: Path) -> pd.DataFrame:
    """Loads and unnests the SCIN dataset metadata."""
    print("Loading SCIN data...")
    scin_df = pd.read_csv(path / "scin_cases.csv")
    
    id_vars = [col for col in scin_df.columns if not col.startswith('image_')]
    
    long_df = pd.melt(scin_df, id_vars=id_vars, value_name='image_path', var_name='image_num')
    long_df = long_df.dropna(subset=['image_path'])
    long_df['image_path'] = long_df.apply(lambda row: path / row['image_path'], axis=1)
    long_df['dataset'] = 'scin'
    
    return long_df

def load_generic_data(path: Path, dataset_name: str) -> pd.DataFrame:
    """Loads a dataset with a simple directory structure."""
    print(f"Loading {dataset_name} data...")
    image_paths = list(path.glob("*.jpg"))
    df = pd.DataFrame({'image_path': image_paths})
    df['dataset'] = dataset_name
    return df

def main(data_dir: Path, output_dir: Path):
    """Main function to load all datasets and save a master list."""
    scin_df = load_scin_data(data_dir / "scin")
    isic_df = load_generic_data(data_dir / "isic", "isic")
    ddi_df = load_generic_data(data_dir / "ddi", "ddi")
    
    master_df = pd.concat([scin_df, isic_df, ddi_df], ignore_index=True)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(output_dir / "master_image_list.csv", index=False)
    print(f"Master image list saved to {output_dir / 'master_image_list.csv'}")
    print(f"Total images loaded: {len(master_df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and consolidate dermatology datasets.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Directory containing the datasets.")
    parser.add_argument("--output_dir", type=Path, default=Path("reports"), help="Directory to save the output CSV.")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
