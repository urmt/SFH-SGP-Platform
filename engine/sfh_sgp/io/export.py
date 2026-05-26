import json
import csv
import io


def export_json(embeddings: list[dict], path: str):
    with open(path, 'w') as f:
        json.dump(embeddings, f, indent=2)


def export_csv(embeddings: list[dict], path: str):
    if not embeddings:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=embeddings[0].keys())
        writer.writeheader()
        writer.writerows(embeddings)


def export_json_str(embeddings: list[dict]) -> str:
    return json.dumps(embeddings, indent=2)
