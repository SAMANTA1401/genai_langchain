import streamlit as st 
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

## Function to get response from LLama 2 model

def getLlamaresponce(input_text,no_words,blog_style):
     
     ## LLama2 model
     llm = CTransformers(model='model\llama-2-7b-chat.ggmlv3.q2_K.bin',
                         model_type = 'llama',
                         config = {'max_new_tokens':256,
                                   'temperature':0.01})

     ## prompt template
     template = """ write a blog {blog_style} job profile for a topic {input_text} within {no_words}. """

     prompt = PromptTemplate(input_variables=["blog_style","input_text",'no_words'],
                             template=template)
     
     ## Gnerate the response from the LLama model
     response = llm(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
     print(response)

     return response    

st.set_page_config(
    page_title="generate blog",
    layout="centered",
    initial_sidebar_state="collapsed",

)

st.header("Generate Blogs ")

input_text = st.text_input("Enter the blog topic")

# createing to more columns for additional 2 fields

col1, col2 = st.columns([5,5])

with col1:
    no_words = st.text_input('No of words')
with col2:
    blog_style = st.selectbox('Writing the blog for',('researcher','data scientist','common people'),index=0)

submit = st.button("Generare")

## final response
if submit:
    st.write(getLlamaresponce(input_text, no_words, blog_style))

