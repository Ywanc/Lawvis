import json, os 
#from summarise import *
#from classify_role import truncate
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity

'''
# BM25
print("BM25")
# LegalBert
legalbert_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
legalbert_model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

# SAILER

# DELTA
tokenizer = AutoTokenizer.from_pretrained("CSHaitao/DELTA_EN")
delta_model = AutoModel.from_pretrained("CSHaitao/DELTA_EN")


with open("processed_cases\\test3.json", encoding='utf-8') as f:
    data = json.load(f)

print("Extracting facts ... ")
legal_facts = "Legal facts: " + extract_facts(truncate(data['roles']['Facts'], 5000))
print("Extracting issues ... ")
legal_issues = "Legal issues: " + extract_issue(truncate(data['roles']['Issue'], 5000))

print(legal_facts)
print(legal_issues)'''

# takes in text and returns normalised DELTA sentence embedding
def get_delta_embeddings(delta_model, delta_tokenizer, text):
    # load DELTA tokenizer and model

    # tokenize and get embeddings for a case
    print("Embedding text ... ")
    inputs = delta_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = delta_model(**inputs)
        
    # get CLS token
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    
    # normalize
    cls_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=1)
    return cls_embedding

def search_facts(query_facts, folderpath="SG_cases\\processed_v3", k=5):

    #delta_tokenizer = AutoTokenizer.from_pretrained("CSHaitao/DELTA_EN")
    #delta_model = AutoModel.from_pretrained("CSHaitao/DELTA_EN")
    
    
    print("retrieving candidate cases ... ")
    # retrieve candidate cases and their embeddings
    files = []
    for filename in os.listdir(folderpath):
        if filename.endswith(".json"):
            files.append(filename)
            
    cand_cases = []
    for file in files:
        with open(f'{folderpath}\\{file}', 'r', encoding='utf-8') as f:
            data = json.load(f)
        cand_cases.append((data, torch.tensor(data['facts_embeddings'])))
    
    query_embeddings = get_bge_embeddings(query_facts)
        
    print("calulating similarities ... ")
    similarities = []
    for i, file_emb in enumerate(cand_cases):
        file, emb = file_emb[0], file_emb[1]
        sim = cosine_similarity(query_embeddings.unsqueeze(0), emb.unsqueeze(0).to(query_embeddings.device)).item()
        similarities.append((file, sim))     
    
    # sort by best scores
    result = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
    return result   

def search_issues(query_issues, folderpath="SG_cases\\processed_v3", k=5):

    print("retrieving candidate cases ... ")
    # retrieve candidate cases and their embeddings
    files = []
    for filename in os.listdir(folderpath):
        if filename.endswith(".json"):
            files.append(filename)
            
    cand_cases = []
    for file in files:
        with open(f'{folderpath}\\{file}', 'r', encoding='utf-8') as f:
            data = json.load(f)
        cand_cases.append((data, torch.tensor(data['issues_embeddings'])))
    
    query_embeddings = get_bge_embeddings(query_issues)
        
    print("calulating similarities ... ")
    similarities = []
    for i, file_emb in enumerate(cand_cases):
        file, emb = file_emb[0], file_emb[1]
        sim = cosine_similarity(query_embeddings.unsqueeze(0), emb.unsqueeze(0).to(query_embeddings.device)).item()
        similarities.append((file, sim))     
    
    # sort by best scores
    result = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
    return result   
    
def get_legalbert_embeddings(text, model_name="nlpaueb/legal-bert-base-uncased"):
    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # tokenize
    print("Embedding text ...")
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # take CLS token
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    
    # normalize
    cls_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=1)
    return cls_embedding

def get_bge_embeddings(text):
    model = SentenceTransformer("BAAI/bge-m3")
    embeddings = model.encode(text, convert_to_tensor=True)
    return embeddings
    
# embeds candidate cases on the spot
def test_search(query_facts, k=5):
    
    #delta_tokenizer = AutoTokenizer.from_pretrained("CSHaitao/DELTA_EN")
    #delta_model = AutoModel.from_pretrained("CSHaitao/DELTA_EN")
    model = SentenceTransformer('BAAI/bge-m3')
    
    #query_embeddings = get_delta_embeddings(delta_model, delta_tokenizer, query_facts)
    query_embeddings = model.encode(query_facts, convert_to_tensor=True)
    
    print("Embedding candidate cases ... ")
    cand_cases = []
    test_cases = ["[2025] SGCA 11", "[2025] SGCA 22", "[2025] SGHC 58", "2024 SGCA 30", "2024 SGCA 58", "2024 SGHC 245", "2024 SGCA 30"]
    for file in test_cases:
        with open(f'SG_cases\\processed_cases_v1\\{file}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        cand_cases.append((data, model.encode(data['elements']['Legal Facts'], convert_to_tensor=True)))
    
    print("calulating similarities ... ")
    similarities = []
    for i, file_emb in enumerate(cand_cases):
        file, emb = file_emb[0], file_emb[1]
        sim = cosine_similarity(query_embeddings.unsqueeze(0), emb.unsqueeze(0)).item()
        similarities.append((file, sim))     
    
    # sort by best scores
    result = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
    return result  
    
    

if __name__ == "__main__":
    
    '''
    print("Loading model ... ")
    delta_tokenizer = AutoTokenizer.from_pretrained("CSHaitao/DELTA_EN")
    delta_model = AutoModel.from_pretrained("CSHaitao/DELTA_EN")
    
    # load preexisting embeddings
    print('Loading candidate case embeddings ... ')
    cand_embeddings = []
    files = []
    for filename in os.listdir("SG_cases\\processed_cases"):
        if filename.endswith(".json"):
            files.append(filename)
    for file in files:
        with open(f'processed_cases\\{file}', 'r', encoding='utf-8') as f:
            data = json.load(f)
        cand_embeddings.append((file, get_delta_embeddings(delta_model, delta_tokenizer, data['elements']['Legal Facts'])))
    
    # Client had disagreement with company, was terminated. Client kept on working, and the company denied ever employing him, saying the Client had never signed any employment contract. 
    
    query_text = input("Case facts: ")
    query_embeddings = get_delta_embeddings(delta_model, delta_tokenizer, query_text)
    
    # cosine simliarity calculation
    print("calulating similarities ... ")
    similarities = []
    for i, file_emb in enumerate(cand_embeddings):
        file, emb = file_emb[0], file_emb[1]
        sim = F.cosine_similarity(query_embeddings, emb).item()
        similarities.append((file, sim))     
    
    # sort by best scores
    sorted_sims = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    for file,sim in sorted_sims:
        print(file)
    
    
    
    
    
    

    
    # normalise for cosine similarity
    #cls_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=1)'''
    
    
    
    
    

