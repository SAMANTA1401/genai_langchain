import streamlit as st 
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os   

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


google_api_key = os.environ['GOOGLE_API_KEY']

# genai.configure(api_key=os.environ['GEMINI_API_KEY'])
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text   

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text=text)
    return chunks 

def get_vector_store(text_chunk):
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key ,model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunk, embedding = embeddings)
    vector_store.save_local("faiss_index")
    # return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to use the context information at the right place
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(google_api_key=google_api_key,model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddigs = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model='models/embedding-001')
    new_db =FAISS.load_local('faiss_index',embeddigs,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question":user_question},
        return_only_outputs=True
    )

    print(response)

    st.write("Replay: ", response["output_text"])


def main():
    st.set_page_config("Chat With multiple PDF")
    st.header("chat with multiple pdf using gemini")

    user_question = st.text_input("Ask a Question from the pdf files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        print("try to upload")
        pdf_docs = st.file_uploader("Upload your pdf files and click on the submit button", accept_multiple_files=True)
        print("uploaded file")
        if st.button("Submit & process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing done!")


if __name__=='__main__':
    main()

