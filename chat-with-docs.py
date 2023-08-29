import streamlit as st
from streamlit_chat import message
import os
import openai
import datetime
import json

from tenacity import retry, wait_random_exponential, stop_after_attempt 

from azure.core.credentials import AzureKeyCredential

from azure.search.documents import SearchClient  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.models import Vector  
from azure.search.documents.indexes.models import (  
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    SearchIndex,  
    SemanticConfiguration,  
    PrioritizedFields,  
    SemanticField,  
    SearchField,  
    SemanticSettings,  
    VectorSearch,  
    HnswVectorSearchAlgorithmConfiguration,  
)  

# Load config values
with open(r'config.json') as config_file:
    config_details = json.load(config_file)

# Azure OpenAI Resources
openai_api_base = config_details['OPENAI_API_BASE']
openai_api_key = config_details["OPENAI_API_KEY"]
openai_api_version = config_details['OPENAI_API_VERSION']

chat_model_deployment = config_details['GPT_MODEL']
embeddings_model_deployment = config_details['EMBEDDING_MODEL']

# Azure Cognitive Search Resources
AZSEARCH_KEY = config_details['AZSEARCH_KEY']
AZSEARCH_ENDPOINT= config_details['AZSEARCH_ENDPOINT']
AZSEARCH_INDEX_NAME = config_details['AZSEARCH_INDEX_NAME']
AZSEARCH_API_VERSION = config_details['AZSEARCH_API_VERSION']

openai.api_base = openai_api_base
openai.api_key = openai_api_key
openai.api_version = openai_api_version
openai.api_type = "azure"

#Establish for connectivity to Azure Cognitive Search throughout
credential = AzureKeyCredential(AZSEARCH_KEY)

# region Query Cognitive Search

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
# Function to generate embeddings for use query
def generate_embeddings(text):
    response = openai.Embedding.create(
        input=text, engine=embeddings_model_deployment)
    embeddings = response['data'][0]['embedding']
    return embeddings

# Pure Vector Search
query = "tools for video analysis"  
  
search_client = SearchClient(AZSEARCH_ENDPOINT, AZSEARCH_INDEX_NAME, credential=credential)
vector = Vector(value=generate_embeddings(query), k=10, fields="contentVector")
  
results = search_client.search(  
    search_text=None,  
    vectors= [vector],
    select=["title", "content", "category"],
)  
  
docs_list = []
  
for result in results:  
    # print(f"Title: {result['title']}")  
    # print(f"Score: {result['@search.score']}")  
    # print(f"Text: {result['content']}")  
    # print(f"Category: {result['category']}\n")  
    docs_list.append(f"{result['content']} [{result['title']}] ^{result['@search.score']}")

print(docs_list)


# region PROMPT SETUP

default_prompt = """
You're an Azure Support Specialist helping customers and partners understand the services available. \
You are given a question and a context. You need to answer the question using only the context. \
Be friendly and conversational, don't sound like a computer! \
The context contains information about a service, with the title of the source document enclosed by [] and the relevancy search score follows a ^ \
Format the information from the three most relevant services into a bulleted list. \
If there are no good matches tell the user you cannot find anything, but they can ask again. \
Always include the document title the information was sourced from at the end of each suggestion enclosed in []. \
Do not show the scores to the user.
"""

system_prompt = st.sidebar.text_area("System Prompt", default_prompt, height=200)
seed_message = {"role": "system", "content": system_prompt}
# endregion

# region SESSION MANAGEMENT
# Initialise session state variables
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "messages" not in st.session_state:
    st.session_state["messages"] = [seed_message]
if "model_name" not in st.session_state:
    st.session_state["model_name"] = []
if "cost" not in st.session_state:
    st.session_state["cost"] = []
if "total_tokens" not in st.session_state:
    st.session_state["total_tokens"] = []
if "total_cost" not in st.session_state:
    st.session_state["total_cost"] = 0.0
# endregion

# region SIDEBAR SETUP

counter_placeholder = st.sidebar.empty()
counter_placeholder.write(
    f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}"
)
clear_button = st.sidebar.button("Clear Conversation", key="clear")

if clear_button:
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["messages"] = [seed_message]
    st.session_state["number_tokens"] = []
    st.session_state["model_name"] = []
    st.session_state["cost"] = []
    st.session_state["total_cost"] = 0.0
    st.session_state["total_tokens"] = []
    counter_placeholder.write(
        f"Total cost of this conversation: £{st.session_state['total_cost']:.5f}"
    )


download_conversation_button = st.sidebar.download_button(
    "Download Conversation",
    data=json.dumps(st.session_state["messages"]),
    file_name=f"conversation.json",
    mime="text/json",
)

# endregion


def generate_response(prompt):
    #st.session_state["messages"].append({"role": "user", "content": prompt})
    # Add the user's search question with the specific context (documents from Azure Search).
    st.session_state["messages"].append({"role":"user", "content":f"Question: {user_input} <context> {docs_list} </context>"})
    
    try:
        completion = openai.ChatCompletion.create(
            engine=chat_model_deployment,
            messages=st.session_state["messages"],
        )
        response = completion.choices[0].message.content
    except openai.error.APIError as e:
        st.write(response)
        response = f"The API could not handle this content: {str(e)}"
    st.session_state["messages"].append({"role": "assistant", "content": response})

    # print(st.session_state['messages'])
    total_tokens = completion.usage.total_tokens
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    return response, total_tokens, prompt_tokens, completion_tokens


st.title("Streamlit ChatGPT Demo")

# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key="my_form", clear_on_submit=True):
        user_input = st.text_area("You:", key="input", height=100)
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        output, total_tokens, prompt_tokens, completion_tokens = generate_response(
            user_input
        )
        st.session_state["past"].append(user_input)
        st.session_state["generated"].append(output)
        st.session_state["model_name"].append(chat_model_deployment)
        st.session_state["total_tokens"].append(total_tokens)

        # from https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/#pricing
        cost = total_tokens * 0.001625 / 1000

        st.session_state["cost"].append(cost)
        st.session_state["total_cost"] += cost


if st.session_state["generated"]:
    with response_container:
        for i in range(len(st.session_state["generated"])):
            message(
                st.session_state["past"][i],
                is_user=True,
                key=str(i) + "_user",
                avatar_style="shapes",
            )
            message(
                st.session_state["generated"][i], key=str(i), avatar_style="identicon"
            )
        counter_placeholder.write(
            f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}"
        )