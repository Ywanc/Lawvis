from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st

# takes in user input, routes it to one of the roles
def llm_route_query(client, model_name, query):
    
    system_prompt = """
    You are an legal expert. Given a user query about a legal case, classify it into ONE of the rhetorical role. 
    
    List of Rhetorical Roles:
    - Facts/Issues: Background information, chronology of events, and issues in dispute. What happened? Who are the parties? What is the dispute about?
    - Courts_reasoning : The judge’s logical analysis, legal principles applied, and explanation of how the law is interpreted. Why did the court decide like this? How does the court analyze the law? What precedents or reasoning are applied?
    - Ruling: "The final decision or holding of the court. What did the court decide? What is the outcome?
    - Invalid: User's query isn't related to the case.
    
    Given a section title and text, classify it into **one of the rhetorical roles above**.
    Always return **only** the rhetorical role exactly as listed above, do not include any explanations.
    """   

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Classify this query: {query}"}
    ]
    
    response = client.chat.completions.create(
        model=model_name, 
        messages=messages, 
        temperature=0,
        max_tokens = 50
    )
    return response.choices[0].message.content.strip()

# takes in case, context and query, generates response
def answer(client, model_name, case, context, question):
    system_prompt = f"You are a legal assistant, handling queries on the legal case {case['title']}. Keep your response short and concise, admit if you are not sure."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Here is the case you must use as reference:\n{context}"},
        {"role": "user", "content": question}
    ]

    print("generating ...")
    response = client.chat.completions.create(
        model=model_name,  # hosted on HF
        messages=messages,
        max_tokens=300,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()

# links query routing and answering
def chat(client, model_name, query, case):
    yield "Analysing query ..."
    role = llm_route_query(client, model_name, query)
    print(role)
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
    
    for chunk in answer(client, model_name, case, context, query).split(" "):
        yield chunk + " "