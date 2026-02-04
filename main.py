import subprocess
import sys
from typing import Union
from pathlib import Path

from explain_with_llm import explain_prefix2
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


if __name__ == '__main__':

    DATASET = "BPIC2012-O"
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
    # STEP 2: Extract attention scores (uncomment if needed)
    # ==========================================================================
    # subprocess.run(
    #     [PY, "get_attention_hooked.py", "--dataset", DATASET],
    #     check=True
    # )

    batch_file = OUTPUTS_DIR / DATASET / "batch_prefixes.txt"
    idx = get_longest_prefix_index_from_txt(batch_file)

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