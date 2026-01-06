#!/usr/bin/env python3
"""
Wrapper script to run next_activity.py with fixed random seeds.
This ensures reproducible results without modifying the original ProcessTransformer code.
"""
import os
import sys
import random
import numpy as np

# Set seeds BEFORE importing tensorflow
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Now import tensorflow and set its seed
import tensorflow as tf
tf.random.set_seed(SEED)

# Import and run the original script
# This is equivalent to running next_activity.py but with seeds set
if __name__ == "__main__":
    # Pass through all command line arguments
    sys.argv[0] = "next_activity.py"
    exec(open("next_activity.py").read())
