from transformers import AutoTokenizer
import transformers
import torch
import os
import pandas as pd
import time

hf_access_token = ''
os.environ["HF_ACCESS_TOKEN"] = hf_access_token

model = "meta-llama/Llama-2-13b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer= tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    token = hf_access_token
)

#pipeline.save_pretrained('llama2-13b-chat-hf')

df = pd.read_csv("test_df.csv")

parag = df['original_text'].to_numpy()
ent1 = df['from_entity_name'].to_numpy()
ent2 = df['to_entity_name'].to_numpy()
ent1_of = df['from_entity_start'].to_numpy()
ent2_of = df['to_entity_start'].to_numpy()
original_labels = df['relation_type'].to_numpy()

entities = []
for par, entity1 , entity2, entity1_offset, entity2_offset in zip(parag, ent1, ent2, ent1_of, ent2_of):
    entities.append([ entity1, entity1_offset, entity2, entity2_offset, par])

prompts = []
for elm in entities:
    prompt = f""" [INST] Generate the class of the relation between {elm[0]} starting in character {elm[1]} and {elm[2]} starting in character {elm[3]} from Classes list based on the input. You have to output the class only. Justification and explanation are prohibited.
    Classes: Increase, Decrease, Stop, Start, Improve, Worsen, Presence, Negative_correlation, Affects, Causes, Complicates, Experiences, Interacts_with, Location_of, Marker/Mechanism, Prevents, Reveals, Treats, Physically_related_to, Part_of, Possible, Associated_with, None.
    Input: {elm[4]} 
    Output: [/INST]
    """
    prompts.append(prompt)

print(f"Number of prompts :  {len(prompts)}")

t = time.time()

sequences = pipeline(
    prompts,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    return_full_text=False,
    temperature=0.1,
    eos_token_id=tokenizer.eos_token_id,
)
generated_text = []
for seq in sequences:
    generated_text.append(seq[0]['generated_text'])

print(f"Time taken: {time.time() - t}")

result_dict = {"paragraphs": parag, "entity1": ent1, "entity2": ent2, "original_label": original_labels,"response": generated_text}

result_df = pd.DataFrame(result_dict)

result_df.to_csv("results/unmasked_zeroshot.csv", index=False)
