import requests, json
from pathlib import Path

train_json_url = "https://raw.githubusercontent.com/federicoruggeri/argscichat_project/refs/heads/main/argscichat_allennlp/argscichat_train_dev/fold_0_train.json"
val_json_url = "https://raw.githubusercontent.com/federicoruggeri/argscichat_project/refs/heads/main/argscichat_allennlp/argscichat_train_dev/fold_0_val.json"
test_json_url = "https://raw.githubusercontent.com/federicoruggeri/argscichat_project/refs/heads/main/argscichat_allennlp/argscichat_train_dev/fold_0_test.json"
output_dir = Path("./data/raw")

output_dir.mkdir(exist_ok=True, parents=True)

train_json_response = requests.get(train_json_url)
train_json = train_json_response.json()
with open(output_dir / "fold_0_train.json", "w") as f:
    json.dump(train_json, f, indent=2)
    
val_json_response = requests.get(val_json_url)
val_json = val_json_response.json()
with open(output_dir / "fold_0_val.json", "w") as f:
    json.dump(val_json, f, indent=2)
    
test_json_response = requests.get(test_json_url)
test_json = test_json_response.json()
with open(output_dir / "fold_0_test.json", "w") as f:
    json.dump(test_json, f, indent=2)