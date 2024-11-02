import streamlit as st

def main():
    st.set_page_config(
        page_title="PDF Chat",
        page_icon=":books:"
    )

    st.header("PDF Chat")
    st.subheader("Ask questions about PDF documents")
    st.chat_input("Ask questions")

    with st.sidebar:
        st.subheader("Documents")
        st.file_uploader("Upload your PDFs here")
        st.button("Process")

if __name__ == "__main__":
    main()