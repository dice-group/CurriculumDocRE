import torch
from transformers import BertTokenizerFast
from src.model import EnhancedCurriculumDocREModel  
import json

MODEL_PATH = "output/results/model_enhanced.pt"
BASE_MODEL_NAME = "bert-base-uncased"  
NUM_RELATIONS = 97  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURRICULUM_STAGE = 3 


tokenizer = BertTokenizerFast.from_pretrained(BASE_MODEL_NAME)
model = EnhancedCurriculumDocREModel(base_model_name=BASE_MODEL_NAME, num_relations=NUM_RELATIONS)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}

model.load_state_dict(new_state_dict)
model.to(DEVICE)
model.eval()

# Prepare input text (single example)
text = """Lark Force was an Australian Army formation established in March 1941 during World War II for service in New Britain and New Ireland. Under the command of Lieutenant Colonel John Scanlan, it was raised in Australia and deployed to Rabaul and Kavieng, aboard SS Katoomba, MV Neptuna and HMAT Zealandia, to defend their strategically important harbours and airfields. The objective of the force, was to maintain a forward air observation line as long as possible and to make the enemy fight for this line rather than abandon it at the first threat as the force was considered too small to withstand any invasion. Most of Lark Force was captured by the Imperial Japanese Army after Rabaul and Kavieng were captured in January 1942. The officers of Lark Force were transported to Japan, however the NCOs and men were unfortunately torpedoed by the USS Sturgeon while being transported aboard the Montevideo Maru. Only a handful of the Japanese crew were rescued, with none of the between 1,050 and 1,053 prisoners aboard surviving as they were still locked below deck.
"""


encoding = tokenizer(
    text,
    padding='max_length',
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
input_ids = encoding["input_ids"].to(DEVICE)
attention_mask = encoding["attention_mask"].to(DEVICE)


entity_markers = None 


with torch.no_grad():
    relation_logits, confidence_logit = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        current_curriculum_stage=CURRICULUM_STAGE,
        entity_markers=entity_markers
    )


probabilities = torch.sigmoid(relation_logits)
predicted_labels = (probabilities > 0.05).long()
confidence_score = torch.sigmoid(confidence_logit)


with open("data/rel_info.json", "r") as f:
    rel_info = json.load(f)


relation2id = {rel_id: idx for idx, rel_id in enumerate(rel_info.keys())}
id2wikidata = {idx: rel_id for rel_id, idx in relation2id.items()}
id2name = {idx: rel_info[rel_id] for rel_id, idx in relation2id.items()}


predicted_indices = predicted_labels[0].nonzero(as_tuple=True)[0].tolist()

predicted_relations = [(id2wikidata[idx], id2name[idx]) for idx in predicted_indices]

print("Predicted Relations:")
for rel_id, name in predicted_relations:
    print(f"  - {rel_id}: {name}")

print("Confidence Score:", confidence_score.item())
