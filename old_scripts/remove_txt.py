import os
import sys

target_dir = sys.argv[1]

for f in os.listdir(target_dir):
    if f.endswith('.txt'):
        os.remove(os.path.join(target_dir, f))
