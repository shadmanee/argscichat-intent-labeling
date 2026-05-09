from pathlib import Path
import pandas as pd

url = "https://raw.githubusercontent.com/federicoruggeri/argscichat_project/refs/heads/main/argscichat_allennlp/argscichat_intents/dataset.csv"
output_dir = Path("./data/raw")

output_dir.mkdir(exist_ok=True, parents=True)

output_path = output_dir / "dataset.csv"

df = pd.read_csv(url)
df.to_csv(output_path, index=False)