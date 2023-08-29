# Azure OpenAI Vector Search with Streamlit App

Combines examples from [Azure Cognitive Search Vector Search Code Sample](https://github.com/Azure/cognitive-search-vector-pr/blob/main/demo-python/code/azure-search-vector-python-sample.ipynb) and [Stremlit Harness](https://github.com/microsoft/az-oai-chatgpt-streamlit-harness) to enable a chatbot that searches over a set of documents to provide responses.

Prerequisites:
1. Azure OpenAI Service with Chat & Embedding models deployed.
2. Azure Congitive Search Service

More details for setting these services up is available in the the first notebook.

The steps to use this are:

1. Use the `01-Create_Load_Index.ipynb` file to create an index on Azure Cognitive Search.

2. Test the index and your prompts using `02-Search_Chat.ipynb`

3. To run the app, use `streamlit run chat-with-docs.py.` This will start the app on port 8501. You can then access the app at http://localhost:8501 

