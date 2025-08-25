import streamlit as st
from infer import *

def show_case(case, sim_score):
    #link = case['download link']
    case_facts = case['elements']['Legal Facts']
    if case_facts.startswith("Please"):
        case_facts = "Not found"
    with st.expander(f"**{case['case_id']} | {case['title']}**"):
        st.markdown(
            f"<div style='font-size: 14px; font-family: 'Source Sans Pro', sans-serif;'>"
            f"<b>Facts Summary:</b><br> {case_facts}<br><br>"
            f"<a href='{case['download link']}' target='_blank'>Download PDF</a>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown(f"Similarity score: {sim_score:.4f}")
    
    
    
st.title("AI Case Search")

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
search_mode = st.radio("Search by:", ["**Facts**", "**Issues**"])

if search_mode == "**Facts**":
    query = st.text_area("**Enter Your case facts:** ", placeholder="E.g. Client was alleged to have illegally brought in foreign workers and faking employment passes.")

    if st.button("Search"):
        if query:
            with st.spinner("Searching ... "):
                results = search_facts(query, k=5) # list of (file, sim_score)
    
            for case, sim_score in results:
                show_case(case, sim_score)

        else:
            st.warning("Please enter your case facts.")

elif search_mode == "**Issues**":
    query = st.text_area("**Enter Your case Issue:** ", placeholder="E.g. The main dispute is whether the Defendants' representations misled the Plaintiff.")

    if st.button("Search"):
        if query:
            with st.spinner("Searching ... "):
                results = search_issues(query, k=5) # list of (file, sim_score)
                
            for case, sim_score in results:
                show_case(case, sim_score)
        else:
            st.warning("Please enter your case Issue.")