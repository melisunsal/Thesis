import processtransformer
import pandas as pd
from datetime import datetime, timedelta
import random
import tensorflow, sys, platform


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
    #create_mock()
    print("machine:", platform.machine())
    print("python :", sys.version)