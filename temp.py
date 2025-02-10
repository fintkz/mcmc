import json

def get_prediction_keys(json_file: str) -> list:
    with open(json_file) as f:
        data = json.load(f)
    return sorted(data['predictions'].keys())

if __name__ == "__main__":
    keys = get_prediction_keys("results/model_results.json")
    print("\n".join(keys))
