import streamlit as st 
import lanchchain.langchain_helper as langchain_helper

st.title("resturant name generator")

cusine = st.sidebar.selectbox("pick a cusine",("indian","Italian","mexican","arabic"))



if cusine:
    response = langchain_helper.generate_restaurant_name_and_items(cusine)
    st.header(response['restaurant_name'].strip())
    menu_items = response['menu_items'].strip().split(",")
    st.write("**menu_items**")
    for item in menu_items:
        st.write("-",item)
