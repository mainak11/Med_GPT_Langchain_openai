import os 
from apikey import apikey 

import streamlit as st 
import langchain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('💊🩺 MED GPT 🩺💊')
prompt = st.text_input('DROP THE MEDICINE NAME..') 

# Prompt templates
comp_template = PromptTemplate(
    input_variables = ['topic'], 
    template='composition of medicine {topic}, describe in bullet points'
)

title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='what is the usage of the medicine {topic}, describe in bullet points'
)

script_template = PromptTemplate(
    input_variables = ['topic'], 
    template='what is the side effect of the medicine {topic}, describe in bullet points'
)


# Memory 
comp_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')


# Llms
llm = OpenAI(temperature=0.9) 
comp_chain = LLMChain(llm=llm, prompt=comp_template, verbose=True, output_key='title', memory=comp_memory)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

# wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt: 
    comp = comp_chain.run(prompt)
    title = title_chain.run(prompt)
    # wiki_research = wiki.run(prompt) 
    script = script_chain.run(prompt)

    st.subheader('👨‍🔬⚗️🔬Composition of '+ prompt + '👨‍🔬⚗️🔬')
    st.write(comp) 

    st.subheader('🩺Usage of '+ prompt + '🩺')
    st.write(title) 

    st.subheader('❌Side effect of '+prompt+ '❌')
    st.write(script) 

    # with st.expander('Usage History'): 
    #     st.info(title_memory.buffer)

    # with st.expander('Side Effect History'): 
    #     st.info(script_memory.buffer)

    # with st.expander('Wikipedia Research'): 
    #     st.info(wiki_research)