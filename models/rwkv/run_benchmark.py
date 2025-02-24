import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from livebench.model.models import Model
from livebench.model.model_adapter import register_model_adapter
from rwkv7_adapter import Rwkv7Adapter

# Register the RWKV-7 adapter
register_model_adapter(Rwkv7Adapter)

# Create model instance
model = Model(
    api_name="models/rwkv/rwkv7-2.9B-world",
    display_name="RWKV-7 2.9B World",
    aliases=["rwkv7"],
    adapter=Rwkv7Adapter()
)

# Run the benchmark
if __name__ == "__main__":
    # Run the benchmark using gen_model_answer.py
    model_path = "models/rwkv/rwkv7-2.9B-world"
    model_id = "rwkv7-2.9B-world"
    
    # Generate model answers
    os.system(f"python livebench/gen_model_answer.py --model-path {model_path} --model-id {model_id} --bench-name live_bench/coding")
    
    # Generate ground truth judgments
    os.system(f"python livebench/gen_ground_truth_judgment.py --model {model_id} --bench-name live_bench/coding")
