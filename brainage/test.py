from dotenv import load_dotenv
load_dotenv()

import matplotlib.pyplot as plt
import os
log_dir = os.getenv("LOG_DIR")
model_dir = os.getenv("MODEL_DIR")
data_dir = os.getenv("DATA_DIR")
print(data_dir)