import tiktoken
from tiktoken import get_encoding
from weaviate_interface import WeaviateClient, WhereFilter
from prompt_templates import question_answering_prompt_series, question_answering_system
from openai_interface import GPT_Turbo
from app_features import (convert_seconds, generate_prompt_series, search_result,
                          validate_token_threshold, load_content_cache, load_data)
from reranker import ReRanker
from loguru import logger 
import streamlit as st
import numpy as np
import sys
import json
import os

# load environment variables
from dotenv import load_dotenv
load_dotenv('./keys.env', override=True)

custom_css = '''
<style>
h1 {
    background-color: #f1f1f1;
}
</style>
'''

#st.markdown(custom_css, unsafe_allow_html=True)
        
 
## PAGE CONFIGURATION
st.set_page_config(page_title="Impact Theory", 
                   page_icon=None, 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items=None)
##############
# START CODE #
##############

## RETRIEVER
api_key = os.environ['WEAVIATE_API_KEY']
openAI_api_key = os.environ['OPENAI_API_KEY']
url = os.environ['WEAVIATE_ENDPOINT']

#instantiate client
# If I want weviate to use a new embedding model (like a fine-tuned one) I should explicitely pass it with the argument model_name_or_path
@st.cache_resource
def get_weaviate_client(api_key, url):
    return WeaviateClient(api_key, url)

client = get_weaviate_client(api_key, url)
# These are the different datasets available in the Weaviate datastore. I can have different chunk sizes or different embedding models.
# We can make them available as a drop down menu on the side bar 
available_classes = sorted(client.show_classes())
logger.info(available_classes)

## RERANKER
reranker = ReRanker(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')

## LLM 
model_name = 'gpt-4-turbo'
# I should pass the openAI api key here
llm = GPT_Turbo(model=model_name, api_key=openAI_api_key)

## ENCODING
# This is the tokenizer, we only need it to count the prompt tokens
encoding = tiktoken.encoding_for_model(model_name)

## INDEX NAME
index_name = 'Impact_theory_minilm_256'

##############
#  END CODE  #
##############
data_path = './data/impact_theory_data.json'
@st.cache_data
def get_data(data_path):
    return load_data(data_path)
data = get_data(data_path)
#creates list of guests for sidebar
guest_list = sorted(list(set([d['guest'] for d in data])))

@st.cache_data
def hybrid_search_cached(query, index_name, alpha, limit, display_properties, guest_filter):
    return client.hybrid_search(query, index_name, alpha=alpha, limit=limit, display_properties=display_properties, where_filter=guest_filter)

@st.cache_data
def rerank_results(hybrid_response, query, top_k):
    return reranker.rerank(hybrid_response, query, top_k=top_k, apply_sigmoid=True)

def main():

    with st.sidebar:
        st.subheader(f"Personalise your search!")
        # index = st.selectbox('Select Index/Class. \n Name convention means: <embedding model>_<chunk size>', options=available_classes, index=None, placeholder='Select Index/Class')
        guest = st.selectbox('Select Guest', options=guest_list, index=None, placeholder='Select Guest')
        alpha_slider = st.slider('Hybrid Search weights: 0 = keyword search, 1 = vector search', 0.00, 1.00, 0.30, step=0.05)
        n_slider = st.slider('Hybrid retrieval hits: lower = less docs considered but quicker, higher = more docs considered but slower', 10, 300, 10, step=10)
        k_slider = st.slider('Re ranker final hits: chunks of context actually passed to the LLM and displayed', 1, 5, 3, step=1)
        llm_temperature_slider = st.slider('LLM temperature: 0 = deterministic, 2 = random', 0.0, 2.0, 0.10, step=0.1)
        
    # Otherwise we get an error from the other functions already coded:
    client.display_properties.append('summary')

    st.image('./assets/impact-theory-logo.png', width=400)
    st.subheader(f"Chat with the Impact Theory podcast: ")
    st.write("Welcome! Here you can ask questions and get answers from the podcast episodes. Just type your question in the box below and hit enter. Enjoy!")
    st.write('\n')
    col1, _ = st.columns([7,3])
    with col1:
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')

        if query:

            # make hybrid call to weaviate
            guest_filter = WhereFilter(path=['guest'], operator='Equal', valueText=guest).todict() if guest else None

            # Replace this call in `main()`
            hybrid_response = hybrid_search_cached(query, index_name, alpha_slider, n_slider, client.display_properties, guest_filter)

            # hybrid_response = client.hybrid_search(query, 
            #                                        index_name, 
            #                                        alpha=alpha_slider, 
            #                                        limit=n_slider, 
            #                                        display_properties=client.display_properties, 
            #                                        where_filter=guest_filter)


            # rerank results
            ranked_response = rerank_results(hybrid_response, query, k_slider)
            
            # ranked_response = reranker.rerank(hybrid_response, 
            #                                   query, 
            #                                   top_k=k_slider, 
            #                                   apply_sigmoid=True)
            
            # validate token count is below threshold
            token_threshold = 4000
            valid_response = validate_token_threshold(ranked_response, 
                                                        question_answering_prompt_series, 
                                                        query=query,
                                                        tokenizer=encoding,
                                                        token_threshold=token_threshold, 
                                                        verbose=True)
            
            # prep for streaming response
            llm_call = True
            st.subheader("Response from Impact Theory (context)")
            if llm_call:
                with st.spinner('Generating Response...'):
                     st.markdown("----")
                     #creates container for LLM response
                     chat_container, response_box = [], st.empty()  
                     # generate LLM prompt
                     prompt = generate_prompt_series(query=query, results=valid_response)

                     try:
                     # execute chat call to LLM
                        for resp in llm.get_chat_completion(prompt=prompt, 
                                                            temperature=llm_temperature_slider,
                                                            max_tokens=400,
                                                            show_response=True,
                                                            stream=True):
                        #inserts chat stream from LLM
                            with response_box:
                                content = resp.choices[0].delta.content
                                if content:
                                    chat_container.append(content)
                                    result = "".join(chat_container).strip()
                                    st.write(f'{result}')
                     except Exception as e:
                        print(e)
                       # continue

            st.subheader("Search Results")
            for i, hit in enumerate(valid_response):
                 col1, col2 = st.columns([7, 3], gap='large')
                 image = hit['thumbnail_url']
                 episode_url = hit['episode_url']
                 title = hit['title']
                 show_length = hit['length']
                 time_string = convert_seconds(show_length)

                 with col1:
                     st.write( search_result(i=i, 
                                             url=episode_url,
                                             guest=hit['guest'],
                                             title=title,
                                             content=hit['content'], 
                                             length=time_string),
                                             unsafe_allow_html=True)
                     st.write('\n\n')
                 with col2:
                      #st.write(f"{episode_url} <img src={image} width='200'></a>", 
                      #           unsafe_allow_html=True)
                      st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)

if __name__ == '__main__':
    main()