import json
from typing import Dict, List, Set
from collections import defaultdict

def analyze_type(value):
    if isinstance(value, dict):
        return "dict", {k: analyze_type(v)[0] for k, v in value.items()}
    if isinstance(value, list):
        if not value:
            return "List[Any]", None
        types = {analyze_type(item)[0] for item in value}
        return f"List[{next(iter(types))}]", analyze_type(value[0])[1]
    return type(value).__name__, None

def generate_pydantic_models(json_file: str) -> str:
    with open(json_file) as f:
        data = json.load(f)

    models = defaultdict(set)
    seen_structures = {}
    
    def hash_structure(struct: dict) -> str:
        return json.dumps(sorted(struct.items()), sort_keys=True)
    
    def process_object(obj: dict, prefix: str = "Base") -> str:
        struct_hash = hash_structure({k: analyze_type(v)[0] for k, v in obj.items()})
        
        if struct_hash in seen_structures:
            return seen_structures[struct_hash]
            
        model_name = prefix
        counter = 1
        while model_name in models:
            model_name = f"{prefix}{counter}"
            counter += 1
            
        seen_structures[struct_hash] = model_name
        
        for key, value in obj.items():
            if isinstance(value, dict):
                nested_model = process_object(value, f"{model_name}_{key.capitalize()}")
                models[model_name].add(f"    {key}: {nested_model}")
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                nested_model = process_object(value[0], f"{model_name}_{key.capitalize()}")
                models[model_name].add(f"    {key}: List[{nested_model}]")
            else:
                type_name, _ = analyze_type(value)
                models[model_name].add(f"    {key}: {type_name}")
        
        return model_name

    root_model = process_object(data)
    
    output = ["from pydantic import BaseModel\nfrom typing import List, Optional\n"]
    
    # Generate models from bottom up
    for model_name, fields in models.items():
        output.append(f"\nclass {model_name}(BaseModel):")
        output.extend(sorted(fields))
    
    return "\n".join(output)

if __name__ == "__main__":
    models = generate_pydantic_models("results/model_results.json")
    print(models)
