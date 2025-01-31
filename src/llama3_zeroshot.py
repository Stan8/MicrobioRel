from transformers import AutoTokenizer
import transformers
import torch
import os
import pandas as pd
import time

hf_access_token = ''
os.environ["HF_ACCESS_TOKEN"] = hf_access_token

model = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer= tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
    token = hf_access_token 
)




terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

#pipeline.save_pretrained('llama2-13b-chat-hf')

df = pd.read_csv("test_df.csv")

parag = df['original_text'].to_numpy()
ent1 = df['from_entity_name'].to_numpy()
ent2 = df['to_entity_name'].to_numpy()
ent1_of = df['from_entity_start'].to_numpy()
ent2_of = df['to_entity_start'].to_numpy()
labels = df['filtered_labels'].to_numpy()
original_labels = df['relation_type'].to_numpy()

entities = []
for par, entity1 , entity2, entity1_offset, entity2_offset in zip(parag, ent1, ent2, ent1_of, ent2_of):
    entities.append([ entity1, entity1_offset, entity2, entity2_offset, classes])


# Initialize an empty list to store prompts
prompts = []

# Iterate over each row in test_df
for elm in entities:
    
    
    messages = [
    {"role": "system", "content": "Generate the class of the relation between the entities. Pick the class from Classes list based on the text. You have to output the class only. Justification and explanation are prohibited."},
    {"role": "user", "content": f" Text: {elm[5]} \n From the Class list: Increase, Decrease, Stop, Start, Improve, Worsen, Presence, Negative_correlation, Affects, Causes, Complicates, Experiences, Interacts_with, Location_of, Marker/Mechanism, Prevents, Reveals, Treats, Physically_related_to, Part_of, Possible, Associated_with, None., what is the relation between {elm[0]} starting in character {elm[1]} and {elm[2]} starting in character {elm[3]} in this text?"},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # Append the constructed prompt
    prompts.append(prompt)

print(f"Number of prompts :  {len(prompts)}")
t = time.time()
sequences = pipeline(
    prompts,
    max_new_tokens=400,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    return_full_text=False,
    eos_token_id=terminators,
)


generated_text = []
for i in range(len(sequences)):
    generated_text.append(sequences[i][0]['generated_text'])

print(f"Time taken: {time.time() - t}")

df['response'] = generated_text

df.to_csv("results/llama3_zeroshot.csv", index=False)
