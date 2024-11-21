
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings

import requests
import pandas as pd
import os

# Define environmental variables for access to Langchain API
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

currURL = "https://datasets-server.huggingface.co/rows"


# Iterate for 5000 times to get the first 5000 rows of the dataset
# Need to iterate by steps of 100 due to API limits and get these rows and append to existing list

def getFinalDataFrame(finalDataset):
    for currStart in range(0, 5001, 100):
        params = {
            "dataset": "sakthisanthosh11/Medication_interaction-1",
            "config": "default",
            "split": "train",
            "offset": currStart,
            "length": 100
        }

        currResp = requests.get(currURL, params=params)

        if currResp.status_code == 200:
            data = currResp.json()

            currListData = data.get('rows', [])
            currentRows = [currR['row'] for currR in currListData]
            finalDataset.extend(currentRows)
        else:
            print("Failed to access LangChain API")
    medDF = pd.DataFrame(finalDataset)
    return medDF


finalMedicationDF = getFinalDataFrame([])


medInfoList = finalMedicationDF['text'].tolist()


# Convert each row in the dataset into a Document object to represent as context
documents = []
for currMedInfo in medInfoList:
    document = Document(page_content=currMedInfo)
    documents.append(document)

# RecursiveCharacterTextSplitter is best to preserve the meaning of each of the medication interaction and ensuring that the interactions between two medications are kept track of
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=24)

currTokenSplits = text_splitter.split_documents(documents)


huggingFaceEmbeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5")

currVectors = Chroma.from_documents(
    documents=currTokenSplits, embedding=huggingFaceEmbeddings)


llm = OllamaLLM(model="llama3")
prompt = hub.pull("rlm/rag-prompt")

retriever = currVectors.as_retriever()


# Combining all of the documents into a single string(each row is separated by a new line for clarity) to process as context for the LLM
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Define query to ask the model about the medication interactions dataset
currQuery = "Tell me about the medications that have the most interactions with other medications."
currAns = rag_chain.invoke(currQuery)
print()
print("Current Response: \n", currAns)
