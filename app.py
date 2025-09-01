import streamlit as st
from infer import *
from chat import *
import time
from openai import OpenAI

st.set_page_config(page_title="Lawvis", initial_sidebar_state="expanded")
hf_token = st.secrets["HF_TOKEN"]

@st.cache_resource
def get_client():
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token,
    )
    model_name = "Qwen/Qwen2-1.5B-Instruct:featherless-ai"
    return client, model_name

client, model_name = get_client()

# show a case in search results
def show_case(case, sim_score):
    def select_case(case):
        st.session_state.selected_case = case
    case_facts = case['elements']['Legal Facts']
    if case_facts.startswith("Please"):
        case_facts = "Not found"
    
    # show a case
    with st.expander(f"**{case['case_id']} | {case['title']}**"):
        
        # user clicks the 'View Case' button
        st.button("View Case", key=f"view_{case['case_id']}", on_click=select_case, args=(case,))
            
        st.markdown(
            f"<div style='font-size: 14px; font-family: 'Source Sans Pro', sans-serif;'>"
            f"<b>Facts Summary:</b><br> {case_facts}<br><br>"
            f"<b>Issues Summary:</b><br> {case['elements']['Legal Issues']}<br><br>"
            f"<a href='{case['download link']}' target='_blank'>Download PDF</a>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown(f"Similarity score: {sim_score:.4f}")

# chatbot sidebar
def chatbot_sidebar(case):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if len(st.session_state.chat_history) == 0:  # If the conversation hasn't started yet
        initial_message = f"""
        👋 Hi there! I'm Lawvis, here to help you out on the case of: 
        
        **{case['title']}**
        
        Feel free to ask me any queries you might have on this case!
        """
        st.session_state.chat_history.append({"role": "assistant", "content": initial_message})

    
    with st.sidebar:
        st.title("Lawvis Case QnA")
        st.write(f"model: {model_name}")
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Get user input
        user_query = st.chat_input("What was the plaintiff's cause of action?")

        # Process user input and generate response
        if user_query:
            # Append user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            # Re-render chat to show user input at the bottom
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_query)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()  # Placeholder for assistant's message
                    spinner_placeholder = st.empty()
                    
                    full_response = ""
                    for update in chat(model=model, tokenizer=tokenizer, case=case, query=user_query):
                        if update.endswith("..."):
                            spinner_placeholder.markdown(f"{update}")
                        else:
                            spinner_placeholder.empty()
                            full_response += update
                            message_placeholder.markdown(full_response + "▌")
                            time.sleep(0.05)
                
                    message_placeholder.markdown(full_response)

                # Append the assistant's final response to the chat history
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})

# case page duh   
def case_page():
    def go_back():
        st.session_state.selected_case = None
        st.session_state.chat_history = []
    case = st.session_state.selected_case
    chatbot_sidebar(case)
    st.button("Back", on_click=go_back)
    st.markdown(f"<h4 style='text-align:center;'>{case['case_id']}</h4>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align:center;'>{case['title']}</h2>", unsafe_allow_html=True)
    
    for section in case['sections']:
        st.markdown(f"**{section['title']}**")
        for para in section['paragraphs']:
            st.write(para)

# home page for search
def search_page():
    
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "results" not in st.session_state:
        st.session_state.results = []
    if "mode" not in st.session_state:
        st.session_state.mode = 0

    st.title("Lawvis.ai")
    st.write("End of Endless Case Search")

    description = """
    An AI-powered legal case search system that matches legal cases based on Facts or Issue similarity.

    Current prototype only contains around 150 Singapore cases from 2024 to 2025. Inaccuries in search may be due to low variety of cases. 
    """

    how_to_use = """
    **Step 1**: Choose either to search by **Facts** or **Issues**

    **Step 2**: Enter your Facts/Issues to search for similar case, press the Search button.

    **Step 3**: The system will return top 5 cases that are most similar to your Facts/Issues.
    """
    with st.expander("Description"):
        st.write(description)
        #st.write("The system only has a limited amount of cases for testing. Inaccuries in search result may be due to the lack of cases.")

    with st.expander("How to use"):
        st.write(how_to_use)
        
    search_mode = st.radio("Search by:", ["**Facts**", "**Issues**"], horizontal=True, index=st.session_state.mode)
    
    if search_mode == "**Facts**":
        st.session_state.mode = 0
        st.session_state.query = st.text_area("**Enter Your case facts:** ", value=st.session_state.query, placeholder="E.g. Client was alleged to have illegally brought in foreign workers and faking employment passes.")

        if st.button("Search"):
            if st.session_state.query:
                with st.spinner("Searching ... "):
                    st.session_state.results = search_facts(st.session_state.query, k=5) # list of (file, sim_score)

            else:
                st.warning("Please enter your case facts.")

    elif search_mode == "**Issues**":
        st.session_state.mode = 1
        st.session_state.query = st.text_area("**Enter Your case Issue:** ", value=st.session_state.query, placeholder="E.g. The main dispute is whether the Defendants' representations misled the Plaintiff.")

        if st.button("Search"):
            if st.session_state.query:
                with st.spinner("Searching ... "):
                    st.session_state.results = search_issues(st.session_state.query, k=5) # list of (file, sim_score)
                    
            else:
                st.warning("Please enter your case Issue.")

    # show previous search results if any
    if st.session_state.results:
        for case, sim_score in st.session_state.results:
            show_case(case, sim_score)     

if __name__ == "__main__":
    # check is any case is clicked
    if "selected_case" not in st.session_state:
        st.session_state.selected_case = None
        
    if st.session_state.selected_case is not None:
        case_page()
    else:
        search_page()