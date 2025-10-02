import os, math, random
from datetime import datetime, timedelta
import pandas as pd
random.seed(42)

# --------- knobs (tweak freely) ----------
N_CASES        = 1200                  # how many cases to generate
MIN_LEN, MAX_LEN = 4, 18               # min/max events per case
ACTIVITIES     = ["Register","Classify","Analyze","Escalate","Wait","Resolve","Close"]
ROLES          = {"Register":"Agent L1",
                  "Classify":"Agent L1",
                  "Analyze":"Agent L2",
                  "Escalate":"Supervisor",
                  "Wait":"Queue",
                  "Resolve":"Agent L2",
                  "Close":"Agent L1"}
# per-activity mean & std (hours) for inter-event time (lognormal-friendly)
DUR_H_MEAN_STD = {
    "Register": (0.1, 0.05),
    "Classify": (0.3, 0.1),
    "Analyze" : (6.0, 2.0),
    "Escalate": (1.5, 0.5),
    "Wait"    : (24.0, 12.0),
    "Resolve" : (8.0, 3.0),
    "Close"   : (0.1, 0.05),
}

# Transition matrix (rows sum ≈ 1). Simple Markovian control-flow with branching & loops.
#        To:   Reg  Clas Anlz Esc  Wait Res  Close
TM = {  # From:
    "Register": [0.0, 0.95, 0.05, 0.00, 0.00, 0.00, 0.00],
    "Classify": [0.0, 0.00, 0.75, 0.10, 0.10, 0.00, 0.05],
    "Analyze" : [0.0, 0.00, 0.10, 0.30, 0.15, 0.40, 0.05],
    "Escalate": [0.0, 0.00, 0.55, 0.00, 0.10, 0.30, 0.05],
    "Wait"    : [0.0, 0.00, 0.40, 0.10, 0.10, 0.30, 0.10],
    "Resolve" : [0.0, 0.00, 0.05, 0.00, 0.05, 0.10, 0.80],
    "Close"   : [0.0, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],  # absorbing
}
NEXT = {act: list(zip(ACTIVITIES, probs)) for act, probs in TM.items()}

def draw_next(curr):
    acts, probs = zip(*NEXT[curr])
    r = random.random()
    cum = 0.0
    for a, p in zip(acts, probs):
        cum += p
        if r <= cum: return a
    return acts[-1]

def lognormal_hours(mu_h, sigma_h):
    # convert mean/std in *hours* to lognormal (mu, sigma in log space)
    # mean = exp(m + s^2/2), var = (exp(s^2)-1)exp(2m+s^2)
    # solve for m,s given mean and std
    mean, std = mu_h, sigma_h
    var = std**2
    s2 = math.log(1 + var/(mean**2)) if mean > 0 and var > 0 else 1e-6
    s  = math.sqrt(max(s2, 1e-9))
    m  = math.log(max(mean, 1e-6)) - s2/2
    # sample in hours
    return random.lognormvariate(m, s)

def gen_case(case_id, start_dt):
    events = []
    t = start_dt
    act = "Register"
    steps = 0
    while True:
        # record current event
        events.append((case_id, act, t, ROLES.get(act, "Unknown")))
        steps += 1
        if act == "Close" or steps >= MAX_LEN:
            break
        # sample next act and inter-event time
        next_act = draw_next(act)
        # tiny loop-avoidance for super-short cases
        if steps < MIN_LEN and next_act == "Close":
            next_act = "Resolve"
        mu, sd = DUR_H_MEAN_STD[next_act]
        dt_hours = max(0.01, lognormal_hours(mu, sd))
        t = t + timedelta(hours=dt_hours)
        act = next_act
    return events

def main():
    os.makedirs("datasets/mockLargerDataset", exist_ok=True)
    rows = []
    base = datetime(2024, 1, 1, 8, 0, 0)
    for cid in range(1, N_CASES + 1):
        start = base + timedelta(minutes=random.randint(0, 60*60))  # scatter starts
        for case_id, act, ts, role in gen_case(cid, start):
            rows.append({"Case ID": case_id,
                         "Activity": act,
                         "Complete Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                         "role": role})
    df = pd.DataFrame(rows).sort_values(["Case ID", "Complete Timestamp"])
    outp = "datasets/mockLargerDataset/mockLargerDataset.csv"
    df.to_csv(outp, index=False)
    print(f"Wrote {len(df)} events across {N_CASES} cases -> {outp}")

if __name__ == "__main__":
    main()
