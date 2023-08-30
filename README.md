# Azure OpenAI Vector Search with Streamlit App

Combines examples from [Azure Cognitive Search Vector Search Code Sample](https://github.com/Azure/cognitive-search-vector-pr/blob/main/demo-python/code/azure-search-vector-python-sample.ipynb) and [Streamlit Harness](https://github.com/microsoft/az-oai-chatgpt-streamlit-harness) to enable a chatbot that searches over a set of documents to provide responses.


# Prerequisites

**Azure Cognitive Search**

Details for creating an Azure Cognitive Search Service are available in [Create an Azure Cognitive Search service in the portal](https://learn.microsoft.com/en-us/azure/search/search-create-service-portal).  


**Azure OpenAI**

To setup an Azure OpenAI Service, please see [Create and deploy an Azure OpenAI Service resource](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal).  


Two types of large language models are used for the document chatbot system and must be deployed in your Azure OpenAI Service: <br/>
1) Similarity [Embeddings](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/concepts/understand-embeddings) model designed for creating the embeddings used for finding similarity between snippets of text.<br/>
2) [GPT-35-Turbo or GPT-4](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/chatgpt?pivots=programming-language-chat-completions) model for the conversational interface <br/>

This example uses the **text-embedding-ada-002** model for similarity embeddings and **gpt-35-turbo** for the conversational functionality. 


# Running the Application

The steps to setup and run the Vector Search chatbot are as follows:

1. Use the `01-Create_Load_Index.ipynb` file to create an index on Azure Cognitive Search.

2. Test the index and your prompts using `02-Search_Chat.ipynb`

3. To run the app, use `streamlit run chat-with-docs.py.` This will start the app on port 8501. You can then access the app at http://localhost:8501 

