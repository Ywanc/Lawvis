import streamlit as st
from infer import *

st.title("AI Case Search")


with st.expander(f"Description"):
    st.write("An AI-powered legal case search system that uses the power of word embeddings to match similar legal cases.\n To use, choose your mode of search, either search by facts . The system will return the top 5 most relevant cases based on the case facts you gave, along with a brief summary of the case.")
    st.write("The system only has a limited amount of cases for testing. Inaccuries in search result may be due to the lack of cases.")

search_mode = st.radio("Search by:", ["Facts", "Issues"])

if search_mode == "Facts":
    query = st.text_area("**Enter Your case facts:** ", placeholder="E.g. Client was alleged to have illegally brought in foreign workers and faking employment passes.")

    if st.button("Search"):
        if query:
            with st.spinner("Searching ... "):
                results = search_facts(query, k=7) # list of (file, sim_score)
                
            for case, sim_score in results:
                link = case['download link']
                with st.expander(f"**{case['case_id']} | {case['title']}**"):
                    st.markdown(
                        f"<div style='font-size: 14px; font-family: 'Source Sans Pro', sans-serif;'>"
                        f"<b>Facts Summary:</b><br> {case['elements']['Legal Facts']}<br><br>"
                        f"<b>Issue Summary:</b><br> {case['elements']['Legal Issues']}<br><br>"
                        f"<a href='{case['download link']}' target='_blank'>Download PDF</a>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"Similarity score: {sim_score:.4f}")

        else:
            st.warning("Please enter your case facts.")

elif search_mode == "Issues":
    query = st.text_area("**Enter Your case Issue:** ", placeholder="E.g. The main dispute is whether the Defendants' representations misled the Plaintiff.")

    if st.button("Search"):
        if query:
            with st.spinner("Searching ... "):
                results = search_issues(query, k=7) # list of (file, sim_score)
                
            for case, sim_score in results:
                link = case['download link']
                with st.expander(f"**{case['case_id']} | {case['title']}**"):
                    st.markdown(
                        f"<div style='font-size: 14px; font-family: 'Source Sans Pro', sans-serif;'>"
                        f"<b>Facts Summary:</b><br> {case['elements']['Legal Facts']}<br><br>"
                        f"<b>Issue Summary:</b><br> {case['elements']['Legal Issues']}<br><br>"
                        f"<a href='{case['download link']}' target='_blank'>Download PDF</a>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"Similarity score: {sim_score:.4f}")

        else:
            st.warning("Please enter your case Issue.")