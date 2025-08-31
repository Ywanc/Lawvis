from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def route_query(query):
    model = ChatOllama(model="qwen3:4b", temperature=0)
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
    
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ('human', "Classify this query: {query}")
    ])
    
    chain = prompt_template | model | StrOutputParser()
    response = chain.invoke({"query": query})
    print(response.split('</think>', 1)[1].strip())
    return response.split('</think>', 1)[1].strip()

def answer(case, context, question):
    chat_model = ChatOllama(model="qwen3:4b", temperature=0.1, num_ctx=20000)

    system_prompt = f"You are a legal assistant, handling queries on the legal case {case['title']}. Keep your response short and concise, **admit if you are not sure**."
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ("system", "Here is the case you must use as reference:\n{context}"),
        ('human', "{question}")
    ])

    chain = prompt_template | chat_model | StrOutputParser()
    response = chain.invoke({"context": context, "question": question})
    return response.split('</think>', 1)[1].strip()

def chat(query, case):
    yield "Analysing query ..."
    role = route_query(query)
    
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
        
    for chunk in answer(case, context, query).split(" "):
        yield chunk + " "