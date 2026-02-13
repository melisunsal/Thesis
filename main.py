import argparse
import subprocess
import sys
from typing import Union
from pathlib import Path

from explain_with_llm_v2 import explain_prefix


def get_longest_prefix_index_from_txt(path: Union[str, Path]) -> int:
    path = Path(path)
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    prefixes = [ln for ln in lines if ln]  # drop empty lines

    if not prefixes:
        raise ValueError(f"No prefixes found in {path}")

    lengths = [len(p.split()) for p in prefixes]
    return max(range(len(lengths)), key=lengths.__getitem__)


def get_nth_longest_prefix_index_from_txt(
        path: Union[str, Path],
        n: int = 0
) -> int:
    """
    n=0 -> longest
    n=1 -> second longest
    n=2 -> third longest
    """
    path = Path(path)
    prefixes = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    if len(prefixes) <= n:
        raise ValueError(f"Not enough prefixes in {path} to get rank {n}")

    lengths = [(i, len(p.split())) for i, p in enumerate(prefixes)]
    lengths.sort(key=lambda x: x[1], reverse=True)

    return lengths[n][0]


def load_prefix_by_index_from_txt(path: Union[str, Path], idx: int) -> list[str]:
    """Convenience: returns the tokenized prefix at idx."""
    path = Path(path)
    prefixes = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return prefixes[idx].split()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the explanation generation pipeline for Transformer-based PPM"
    )
    parser.add_argument(
        "--dataset", type=str, default="BPIC2012-O",
        help="Dataset name (default: BPIC2012-O)"
    )
    parser.add_argument(
        "--prefix_index", type=int, default=None,
        help="Prefix index to explain. If not specified, uses the longest prefix."
    )
    parser.add_argument(
        "--generate_batch", action="store_true",
        help="Generate new batch by running get_attention_hooked.py first"
    )
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    DATASET = args.dataset
    OUTPUTS_DIR = Path("outputs")

    PY = sys.executable

    # ==========================================================================
    # STEP 1: Train model (uncomment if needed)
    # ==========================================================================
    # subprocess.run(
    #     [PY, "next_activity.py",
    #      "--dataset", DATASET,
    #      "--epochs", str(15)],
    #     check=True
    # )

    # ==========================================================================
    # STEP 2: Extract attention scores (run with --generate_batch flag)
    # ==========================================================================
    if args.generate_batch:
        print("Generating new batch prefixes...")
        subprocess.run(
            [PY, "get_attention_hooked.py", "--dataset", DATASET],
            check=True
        )

    batch_file = OUTPUTS_DIR / DATASET / "batch_prefixes.txt"

    # Use provided prefix_index or default to longest prefix
    if args.prefix_index is not None:
        idx = args.prefix_index
        print(f"Using provided prefix index: {idx}")
    else:
        idx = get_longest_prefix_index_from_txt(batch_file)
        print(f"Using longest prefix index: {idx}")

    # # second longest prefix
    # idx_second = get_nth_longest_prefix_index_from_txt(batch_file, n=1)
    #
    # # second longest prefix
    # idx_third = get_nth_longest_prefix_index_from_txt(batch_file, n=2)

    # ==========================================================================
    # STEP 3: Run visualization and SHAP computation
    # ==========================================================================
    for i in [idx]:
        subprocess.run(
            [PY, "visualizeAttention/visualization.py",
             "--dataset", DATASET,
             "--prefix_index", str(i)],
            check=True
        )

        subprocess.run(
            [PY, "shapley_ppm_deletion_only_shap_pkg.py",
             "--dataset", DATASET,
             "--prefix_index", str(i)],
            check=True
        )
    #
    #     subprocess.run(
    #         [PY, "shapley_ppm_pad_replace_shap_pkg.py",
    #          "--dataset", DATASET,
    #          "--prefix_index", str(i)],
    #         check=True
    #     )

    # ==========================================================================
    # STEP 4: Run verification experiments
    # ==========================================================================
    # for i in [idx, idx_second]:
    #     subprocess.run(
    #         [PY, "shap_experiments_sensible.py",
    #          "--dataset", DATASET,
    #          "--prefix_index", str(i),
    #          "--experiment", "all"],
    #         check=True
    #     )

    # ==========================================================================
    # STEP 5: Generate LLM Explanations
    # ==========================================================================
    print("\n" + "=" * 80)
    print("GENERATING LLM EXPLANATIONS")
    print("=" * 80 + "\n")

    for i in [idx]:
        print(f"\n--- Explanation for prefix_index={i} ---\n")

        # GPT version (requires OPENAI_API_KEY environment variable)
        try:
            result = explain_prefix(
                dataset_name=DATASET,
                out_dir="./outputs",
                prefix_index=i,
                backend="gpt",
                llm_model_name=None,  # uses OPENAI_MODEL_NAME env var or default
            )

            print("EXPLANATION:")
            print("-" * 40)
            print(result.get("explanation"))
            print("-" * 40)
            print(f"Confidence: {result.get('confidence_level')}")

            # Print alignment analysis
            alignment = result.get("alignment_analysis", {})
            if alignment:
                print(f"Signal alignment: {alignment.get('alignment_strength', 'N/A')}")
                print(f"Aligned events: {len(alignment.get('aligned_events', []))}")
                if alignment.get('aligned_events'):
                    for evt in alignment['aligned_events']:
                        print(f"  - Event {evt['position'] + 1} ({evt['activity']}): "
                              f"attn={evt['attention_weight']:.0%}, shap={evt['shap_value']:+.2f}")

            print()

        except Exception as e:
            print(f"Error generating explanation: {e}")
            print("Make sure OPENAI_API_KEY is set in your environment.\n")

    # ==========================================================================
    # Alternative: Local LLM (no API key needed)
    # ==========================================================================
    #
    # print("\n" + "=" * 80)
    # print("GENERATING LLM EXPLANATIONS")
    # print("=" * 80 + "\n")
    #
    # for i in [idx]:
    #     print(f"\n--- Explanation for prefix_index={i} ---\n")
    #
    #     # GPT version (requires OPENAI_API_KEY environment variable)
    #     try:
    #         result = explain_prefix(
    #             dataset_name=DATASET,
    #             out_dir="./outputs",
    #             prefix_index=i,
    #             backend="local",  # Uses TinyLlama by default
    #             llm_model_name=None,
    #         )
    #
    #         print("EXPLANATION:")
    #         print("-" * 40)
    #         print(result.get("explanation"))
    #         print("-" * 40)
    #         print(f"Confidence: {result.get('confidence_level')}")
    #
    #         # Print alignment analysis
    #         alignment = result.get("alignment_analysis", {})
    #         if alignment:
    #             print(f"Signal alignment: {alignment.get('alignment_strength', 'N/A')}")
    #             print(f"Aligned events: {len(alignment.get('aligned_events', []))}")
    #             if alignment.get('aligned_events'):
    #                 for evt in alignment['aligned_events']:
    #                     print(f"  - Event {evt['position'] + 1} ({evt['activity']}): "
    #                           f"attn={evt['attention_weight']:.0%}, shap={evt['shap_value']:+.2f}")
    #
    #         print()
    #
    #     except Exception as e:
    #         print(f"Error generating explanation: {e}")