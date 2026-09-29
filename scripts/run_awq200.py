"""AWQ 200-item GSM8K evaluation (background run)."""
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from flipgate.scorers.gsm8k import GSM8KScorer
from flipgate.store import ResultsStore, generate_run_id

print("Loading GSM8K...", flush=True)
gsm8k = load_dataset("openai/gsm8k", "main", split="test")
items = [{"question": it["question"], "answer": it["answer"]} for it in gsm8k][:200]
print(f"Using {len(items)} items", flush=True)

print("Loading AWQ model...", flush=True)
model_path = "data/models/Qwen--Qwen2.5-3B-Instruct-AWQ"
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.float16, device_map="cuda",
    pad_token_id=tokenizer.pad_token_id,
)
print("Model loaded", flush=True)

scorer = GSM8KScorer()
store = ResultsStore("data/results")
run_id = generate_run_id("Qwen2.5-3B-AWQ", "hf_generate", "gsm8k", 1)
print(f"Run ID: {run_id}", flush=True)

correct = 0
start = time.time()
for i, item in enumerate(items):
    messages = [{"role": "user", "content": item["question"]}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    score = scorer.score(item["question"], response, {"answer": item["answer"]})
    if score == 1.0:
        correct += 1
    store.append_item(run_id, "gsm8k", f"item_{i:04d}", item["question"], response, score)
    if (i + 1) % 20 == 0:
        print(f"Progress: {i+1}/200, accuracy: {correct/(i+1):.1%}", flush=True)

elapsed = time.time() - start
accuracy = correct / len(items)
store.append_run_metadata(
    run_id, "Qwen2.5-3B-Instruct-AWQ", "hf_generate", "5.17.0", "gsm8k", "test",
    batch_size=1, extra={"accuracy": accuracy, "num_items": len(items), "elapsed": elapsed},
)
print(f"Done! {accuracy:.1%} ({correct}/{len(items)}) in {elapsed:.1f}s", flush=True)
print(f"Run ID: {run_id}", flush=True)
