"""
VLM Annotation Script - Phase 2

This script uses a state-of-the-art Vision Language Model to generate
quality labels for all images in the master list. This represents a novel
knowledge distillation approach for dermatologic image quality assessment.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import argparse
from tqdm import tqdm
import time
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.diqa.vlm_client import create_vlm_client
from scripts.utils.run_manager import RunManager


def annotate_images(master_list_path: Path, output_dir: Path, provider: str = "anthropic",
                    model: str = None, batch_size: int = 100, rate_limit_delay: float = 1.0,
                    resume: bool = True, force: bool = False):
    """
    Main function to annotate all images using VLM.

    Args:
        master_list_path: Path to master_image_list.csv
        output_dir: Directory to save VLM labels
        provider: VLM provider ('openai' or 'anthropic')
        model: Specific model name (optional, uses provider defaults)
        batch_size: Number of images to process before saving checkpoint
        rate_limit_delay: Seconds to wait between API calls
        resume: Whether to resume from existing checkpoint
    """
    run_manager = RunManager()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "vlm_labels.csv"
    checkpoint_path = output_dir / "vlm_labels_checkpoint.csv"
    metadata_path = output_dir / "vlm_annotation_metadata.json"

    print(f"Loading master image list from {master_list_path}...")
    df = pd.read_csv(master_list_path)
    df['image_path_str'] = df['image_path'].astype(str)
    unique_targets = df['image_path_str'].nunique()
    print(f"Loaded {len(df)} images across {df['dataset'].nunique()} datasets")

    if not force and output_path.exists():
        try:
            cached_df = pd.read_csv(output_path)
            cached_unique = cached_df['image_path'].astype(str).nunique()
            if cached_unique >= unique_targets:
                print(f"Cached VLM annotations already cover {cached_unique} images (>= target {unique_targets}).")
                print("Skipping annotation. Use --force to recompute from scratch.")
                return
            else:
                print(f"Cached VLM annotations cover {cached_unique}/{unique_targets} images; continuing to annotate remaining samples.")
        except Exception as exc:
            print(f"Warning: failed to read cached VLM labels ({exc}). Will recompute.")

    if force:
        resume = False
        archived = run_manager.archive_previous_outputs([
            "vlm_labels.csv",
            "vlm_labels_checkpoint.csv",
            "vlm_annotation_metadata.json"
        ])
        if archived:
            print(f"Archived {len(archived)} previous output(s) to {run_manager.old_dir}")
        if checkpoint_path.exists():
            checkpoint_path.unlink()
    elif not resume:
        archived = run_manager.archive_previous_outputs([
            "vlm_labels.csv",
            "vlm_labels_checkpoint.csv",
            "vlm_annotation_metadata.json"
        ])
        if archived:
            print(f"Archived {len(archived)} previous output(s) to {run_manager.old_dir}")

    completed_paths = set()
    if resume and checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint_df = pd.read_csv(checkpoint_path)
        completed_paths = set(checkpoint_df['image_path'].tolist())
        print(f"Already processed: {len(completed_paths)} images")
        results_df = checkpoint_df
    else:
        results_df = pd.DataFrame()

    # Create VLM client
    print(f"Initializing {provider} VLM client...")
    client_kwargs = {}
    if model:
        client_kwargs['model'] = model

    vlm_client = create_vlm_client(provider=provider, **client_kwargs)
    print(f"Using model: {vlm_client.model}")

    # Filter to unprocessed images
    to_process = df[~df['image_path_str'].isin(completed_paths)].copy()
    print(f"Remaining to process: {len(to_process)} images\n")

    if len(to_process) == 0:
        print("All images already processed!")
        return

    # Process images
    results = []
    start_time = time.time()

    print("Starting VLM annotation...")
    print("=" * 70)

    for idx, row in tqdm(to_process.iterrows(), total=len(to_process), desc="Annotating"):
        image_path = Path(row['image_path'])

        try:
            # Assess quality using VLM
            assessment = vlm_client.assess_quality(image_path)

            result = {
                'image_path': str(image_path),

                # Main decision and score
                'decision': assessment.decision,
                'quality_score': assessment.quality_score,

                # Individual criterion scores
                'sharpness': assessment.sharpness,
                'exposure': assessment.exposure,
                'glare': assessment.glare,
                'artifacts_obstruction': assessment.artifacts_obstruction,
                'framing_primary_lesion_visible': assessment.framing_primary_lesion_visible,
                'lighting_uniformity': assessment.lighting_uniformity,
                'color_balance': assessment.color_balance,
                'noise_compression': assessment.noise_compression,

                # Critical failures and defects
                'critical_failures': '|'.join(assessment.critical_failures) if assessment.critical_failures else '',
                'defects_detected': '|'.join([d.type for d in assessment.defects]) if assessment.defects else '',
                'defects_localized': '|'.join([
                    f"{d.type}:{d.bbox_norm}" if d.bbox_norm else d.type
                    for d in assessment.defects
                ]) if assessment.defects else '',

                # Image characteristics
                'modality': assessment.modality,
                'polarization': assessment.polarization if assessment.polarization else '',

                # Analysis flags
                'skin_tone_bin': assessment.skin_tone_bin,
                'phi_present': assessment.phi_present,

                # ENHANCED: Uncertainty handling (NEW)
                'uncertain_criteria': '|'.join(assessment.uncertain_criteria) if assessment.uncertain_criteria else '',
                'requires_human_review': assessment.requires_human_review,
                'ambiguity_reason': assessment.ambiguity_reason if assessment.ambiguity_reason else '',

                # Legacy fields for backward compatibility
                'quality_label': assessment.quality_label,
                'vlm_confidence': assessment.confidence,
                'vlm_reasoning': assessment.reasoning,
                'vlm_raw_response': assessment.raw_response,
                'processing_timestamp': datetime.now().isoformat()
            }

            results.append(result)

            # Rate limiting
            time.sleep(rate_limit_delay)

        except Exception as e:
            print(f"\nError processing {image_path}: {e}")
            result = {
                'image_path': str(image_path),

                # Main decision and score
                'decision': 'ERROR',
                'quality_score': None,

                # Individual criterion scores
                'sharpness': None,
                'exposure': None,
                'glare': None,
                'artifacts_obstruction': None,
                'framing_primary_lesion_visible': None,
                'lighting_uniformity': None,
                'color_balance': None,
                'noise_compression': None,

                # Critical failures and defects
                'critical_failures': '',
                'defects_detected': '',
                'defects_localized': '',

                # Image characteristics
                'modality': '',
                'polarization': '',

                # Analysis flags
                'skin_tone_bin': '',
                'phi_present': None,

                # ENHANCED: Uncertainty handling (NEW)
                'uncertain_criteria': '',
                'requires_human_review': None,
                'ambiguity_reason': '',

                # Legacy fields for backward compatibility
                'quality_label': None,
                'vlm_confidence': None,
                'vlm_reasoning': f"ERROR: {str(e)}",
                'vlm_raw_response': '',
                'processing_timestamp': datetime.now().isoformat()
            }
            results.append(result)

        # Save checkpoint every batch_size images
        if len(results) > 0 and len(results) % batch_size == 0:
            batch_df = pd.DataFrame(results)
            if len(results_df) > 0:
                combined_df = pd.concat([results_df, batch_df], ignore_index=True)
            else:
                combined_df = batch_df

            combined_df.to_csv(checkpoint_path, index=False)
            print(f"\nCheckpoint saved: {len(combined_df)} total images processed")
            print(f"Estimated cost so far: ${vlm_client.total_cost:.2f}")

    # Final save
    if len(results) > 0:
        final_df = pd.DataFrame(results)
        if len(results_df) > 0:
            final_df = pd.concat([results_df, final_df], ignore_index=True)

        final_df.to_csv(output_path, index=False)
        print(f"\n{'=' * 70}")
        print(f"VLM annotation complete!")
        print(f"Total images processed: {len(final_df)}")
        print(f"Output saved to: {output_path}")

        # Calculate statistics
        valid_labels = final_df[final_df['decision'].isin(['ACCEPT', 'ACCEPT_WITH_ISSUES', 'REJECT'])]
        if len(valid_labels) > 0:
            accept_rate = valid_labels['decision'].isin(['ACCEPT', 'ACCEPT_WITH_ISSUES']).mean()
            avg_quality_score = valid_labels['quality_score'].mean()
            
            # Additional statistics for new format
            accept_strict_rate = (valid_labels['decision'] == 'ACCEPT').mean()
            accept_with_issues_rate = (valid_labels['decision'] == 'ACCEPT_WITH_ISSUES').mean()
            reject_rate = (valid_labels['decision'] == 'REJECT').mean()
            
            # Score distribution
            avg_sharpness = valid_labels['sharpness'].mean()
            avg_exposure = valid_labels['exposure'].mean()
            critical_failure_rate = (valid_labels['critical_failures'] != '').mean()

            print(f"\nAnnotation Statistics:")
            print(f"- Valid labels: {len(valid_labels)} / {len(final_df)}")
            print(f"- Overall accept rate: {accept_rate:.1%}")
            print(f"  • ACCEPT: {accept_strict_rate:.1%}")
            print(f"  • ACCEPT_WITH_ISSUES: {accept_with_issues_rate:.1%}")
            print(f"  • REJECT: {reject_rate:.1%}")
            print(f"- Average quality score: {avg_quality_score:.2f}")
            print(f"- Average sharpness: {avg_sharpness:.2f}")
            print(f"- Average exposure: {avg_exposure:.2f}")
            print(f"- Critical failure rate: {critical_failure_rate:.1%}")
            print(f"- Total API requests: {vlm_client.request_count}")
            print(f"- Total cost: ${vlm_client.total_cost:.2f}")

        # Save metadata
        elapsed_time = time.time() - start_time
        metadata = {
            'provider': provider,
            'model': vlm_client.model,
            'total_images': len(final_df),
            'valid_labels': len(valid_labels),
            'accept_rate': float(accept_rate) if len(valid_labels) > 0 else None,
            'accept_strict_rate': float(accept_strict_rate) if len(valid_labels) > 0 else None,
            'accept_with_issues_rate': float(accept_with_issues_rate) if len(valid_labels) > 0 else None,
            'reject_rate': float(reject_rate) if len(valid_labels) > 0 else None,
            'average_quality_score': float(avg_quality_score) if len(valid_labels) > 0 else None,
            'average_sharpness': float(avg_sharpness) if len(valid_labels) > 0 else None,
            'average_exposure': float(avg_exposure) if len(valid_labels) > 0 else None,
            'critical_failure_rate': float(critical_failure_rate) if len(valid_labels) > 0 else None,
            'total_requests': vlm_client.request_count,
            'total_cost_usd': vlm_client.total_cost,
            'processing_time_seconds': elapsed_time,
            'timestamp': datetime.now().isoformat()
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nMetadata saved to: {metadata_path}")
        print(f"Processing time: {elapsed_time / 60:.1f} minutes")

        # Log run
        run_manager.log_run(
            phase="02_vlm_annotation",
            status="completed",
            inputs={"master_list": str(master_list_path), "provider": provider, "model": vlm_client.model},
            outputs={"vlm_labels": str(output_path), "metadata": str(metadata_path)},
            metadata=metadata
        )

        # Clean up checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print("Checkpoint file removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate VLM quality labels for dermatology images"
    )
    parser.add_argument(
        "--master_list_path",
        type=Path,
        default=Path("reports/master_image_list.csv"),
        help="Path to master image list CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reports"),
        help="Directory to save VLM labels"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="google",
        choices=["openai", "anthropic", "google", "mock"],
        help="VLM provider to use (mock for testing without API costs)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model name (optional, uses provider defaults)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of images to process before saving checkpoint"
    )
    parser.add_argument(
        "--rate_limit_delay",
        type=float,
        default=1.0,
        help="Seconds to wait between API calls"
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Start from scratch, ignoring any existing checkpoint"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute annotations even if cached results exist"
    )

    args = parser.parse_args()

    annotate_images(
        master_list_path=args.master_list_path,
        output_dir=args.output_dir,
        provider=args.provider,
        model=args.model,
        batch_size=args.batch_size,
        rate_limit_delay=args.rate_limit_delay,
        resume=not args.no_resume,
        force=args.force
    )
