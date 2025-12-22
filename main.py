import processtransformer
import pandas as pd
from datetime import datetime, timedelta
import random
from explain_with_llm import explain_prefix


def create_mock():
    # Create 5 synthetic cases with 3–5 activities each
    activities = ["Register", "Analyze", "Resolve", "Close"]
    rows = []
    case_id = 1

    for _ in range(5):  # 5 cases
        random_hours = random.randint(3, 5)
        start_time = datetime(2025, 1, 1, 8, 0, 0)
        for i in range(random_hours):
            rows.append({
                "Case ID": case_id,
                "Activity": activities[i % len(activities)],
                "Complete Timestamp": (start_time + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
            })
        case_id += 1

    df = pd.DataFrame(rows)
    df.to_csv("datasets/mockdataset/mockdata.csv", index=False)
    print(df.head(10))


if __name__ == '__main__':

    # # GPT version
    # exp_gpt = explain_prefix(
    #     dataset_name="BPIC2012-O",
    #     out_dir="./outputs",
    #     prefix_index=10,
    #     backend="gpt",
    #     llm_model_name=None,  # will use OPENAI_MODEL_NAME or default
    # )
    # print(exp_gpt)

    # LLMama version (once you have the endpoint)
    exp_llmama = explain_prefix(
        dataset_name="BPIC2012-O",
        out_dir="./outputs",
        prefix_index=10,
        backend="llmama",
        llm_model_name=None,  # will use LLMAMA_MODEL_NAME or default
    )
    print(exp_llmama)
