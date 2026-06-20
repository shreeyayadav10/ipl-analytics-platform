import os

# Create folder if not present
os.makedirs("data/raw", exist_ok=True)

# Download IPL dataset
os.system(
    "kaggle datasets download "
    "-d patrickb1912/ipl-complete-dataset-20082020 "
    "-p data/raw --unzip"
)

print("Dataset downloaded successfully.")