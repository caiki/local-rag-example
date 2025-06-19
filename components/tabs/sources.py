import streamlit as st

from components.tabs.local_files import local_files



def sources():
    st.title("Directly import your data")
    st.caption("Convert your data into embeddings for utilization during chat")
    st.write("")

    with st.expander("💻 &nbsp; **Local Files**", expanded=False):
        local_files()
