from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import openai


import os
# openai_api_key = os.environ['OPENAI_API_KEY']


llm = OpenAI(temperature=0.7)
# llm = openai.Completion.create(
#     temperature = 0.6
# )






def generate_restaurant_name_and_items(cusine):
    # Chain 1: Resturant Name
    prompt_template_name = PromptTemplate(
    input_variables = ['cusine'],
    template = "I want to open a restaurant for {cusine} food. Suggest a fency name for this."
    )
    name_chain = LLMChain(llm=llm, prompt= prompt_template_name, output_key="resturant_name")

    promt_template_items = PromptTemplate(
        input_variables = ['resturant_name'],
        template = "Suggest some menu items for {resturant_name} . Return it as as a comma separated."
    )

    menu_items_chain = LLMChain(llm=llm, prompt=promt_template_items, output_key="menu_items")


    chain = SequentialChain(
        chains=[name_chain, menu_items_chain],
        input_variables = ['cusine'],
        output_variables = ['resturant_name','menu_items']
    )

    response = chain({'cusine':cusine})

    return response


if __name__ == "__main__":
    print(generate_restaurant_name_and_items('indian'))

    # promt_template_name = PromptTemplate(

    # )
    # return{
    #     'restaurant_name' : 'curry delight',
    #     'menu_items': 'samosa,paneer tikka',

    # }