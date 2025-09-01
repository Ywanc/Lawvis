from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st

@st.cache_resource
def load_model():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"
    print(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def route_query(model, tokenizer, query):
    system_prompt = """
    You are an legal expert. Given a user query about a legal case, classify it into ONE of the rhetorical role. 
    
    List of Rhetorical Roles:
    - Facts/Issues: The key legal or factual dispute and the factual background leading to it, including how the dispute arose, actions taken by the parties, identities of relevant parties, procedural history, and prior court reasoning or rulings.
    - Courts_reasoning: The court’s analysis of legal issues, arguments from parties, and application of law. Also includes discussion of legal definitions, rules, and interpretation.
    - Ruling: The **current** court's decision or conclusion. First identify the current court handling this case, if the ruling is by any lower court, then label is as "Facts".
    - Invalid: User's query isn't related to the case.
    
    Given a section title and text, classify it into **one of the rhetorical roles above**.
    Always return **only** the rhetorical role exactly as listed above, do not include any explanations.
    """   

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Classify this query: {query}"}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def answer(model, tokenizer, case, context, question):
    system_prompt = f"You are a legal assistant, handling queries on the legal case {case['title']}. Keep your response short and concise, **admit if you are not sure**."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Here is the case you must use as reference:\n{context}"},
        {"role": "user", "content": f"{question}"}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def chat(model, tokenizer, query, case):
    yield "Analysing query ..."
    role = route_query(model, tokenizer, query)
    
    yield "Getting relevant context ..."
    context = case['roles'].get(role, "")
    invalid_flag = False
    if not context:
        context = "No relevant context, prompt the user to ask questions about the case."
        invalid_flag = True
    
    if invalid_flag:
        yield "thinking ..."
    else:
        yield f"Analyzing {role} ..."
        
    for chunk in answer(model, tokenizer, case, context, query).split(" "):
        yield chunk + " "
    