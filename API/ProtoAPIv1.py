# %%
#%pip install "transformers" "accelerate" "langchain" "einops"
#%pip install chromadb
#%pip install langchain
#%pip install sentence_transformers


# %%
import dotenv
import time
import re
import os
import sys
import torch
from transformers import pipeline
from transformers import AutoTokenizer , AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
#loading dotenv file containing path links to databases
load_dotenv()

# Allow all origins



# app = FastAPI()


##
# importing settings
from fastapi.logger import logger
from pydantic import BaseSettings


class Settings(BaseSettings):
    # ... The rest of our FastAPI settings

    BASE_URL = "http://localhost:8000/docs"
    USE_NGROK = os.environ.get("USE_NGROK", "False") == "True"


settings = Settings()


def init_webhooks(base_url):
    # Update inbound traffic via APIs to use the public-facing ngrok URL
    pass
 

# Initialize the FastAPI app for a simple web server
app = FastAPI()

if settings.USE_NGROK:
    # pyngrok should only ever be installed or initialized in a dev environment when this flag is set
    from pyngrok import ngrok

    # Get the dev server port (defaults to 8000 for Uvicorn, can be overridden with `--port`
    # when starting the server
    port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else 8000

    # Open a ngrok tunnel to the dev server
    public_url = ngrok.connect(port).public_url
    logger.info("ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))

    # Update any base URLs or webhooks to use the public ngrok URL
    settings.BASE_URL = public_url
    init_webhooks(public_url)

# ... Initialize routers and the rest of our app
##

startmodel = time.time()

tokenizer = AutoTokenizer.from_pretrained("Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("Llama-2-7b-chat-hf",device_map="auto")
endmodel = time.time()
print(f"Loading model took: {round(endmodel-startmodel,2)} s")


# load model pipeline
generate_text = pipeline("text-generation",model=model, tokenizer = tokenizer,torch_dtype=torch.float16,
                         trust_remote_code=True, device_map= "auto", 
                         do_sample=True,eos_token_id = tokenizer.eos_token_id,
                         max_new_tokens = 512,)


# %%
# Hugging Face model pipeline
from langchain.llms import HuggingFacePipeline

hf_pipeline = HuggingFacePipeline(pipeline=generate_text,model_kwargs={'temperature':0})

# %%
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings

# %%
# Text Embedding Model
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/gtr-t5-base')

# %%
# Vector Database configurations

# %%
# Vector Database instance
# persist_directory = "embeddings" # add path to specific vector database




# %%
# Sample Run on VectorDB for context retrieval
# query = "What are the placement statistics?"
# docs = db.similarity_search_with_score(query)
# context = docs[0][0].page_content
# context[:]

# %%
# Prompt Template 
from langchain import PromptTemplate, LLMChain
prompt_template = ''' 
[INST]  
<<SYS>>:You are a Chatbot specifically designed to respond to user's questions based on the context in 100 words. 
If the question is not related to the context , respond with "I do not know , please rephrase the question".
<</SYS>>
Answer the Question:"{question}" based on the Context:"{context}".
[/INST]
'''

# prompt chain
prompt_with_context = PromptTemplate(
input_variables=["question", "context"],
template= prompt_template)

# %%
# LLM custom Chain for question answering 
#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)

# %%
import time

#engaging stack to store path/multiple paths
# stack = []
# stack.append(persist_directory)
new_path=0

@app.post(f'/api/setpath',response_model=dict)
async def setpath(path_number:str):
    path = os.getenv(path_number)
    os.environ["DB_Path"] = path
    return {"path set to":os.getenv("DB_Path")}
    print(persist_directory)
   

 
    




@app.post('/api/askquery', response_model=dict)
async def askquery(question: str):
  
  

  s1=time.time()
  try:
     while True:
        #Whenever bot starts , cuda set to 1(Either set own value) , comment out the next line for parallel processing
        # os.system('export CUDA_VISIBLE_DEVICES=1')
        # Get the user query 
        query = question
        # if query == "exit":
        #     break
        if query.strip() == "":
            continue

        #Chroma Settings initialised everytime the site is changed
        CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=os.getenv("DB_Path"),
        anonymized_telemetry=False
)

        # Get the answer from the chain
        start = time.time()
        db = Chroma(persist_directory=os.getenv("DB_Path"), embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        
        # similarity search from vectorDB to get context
        docs = db.similarity_search_with_score(query,k=2)
        #print(docs)
        time_ret= time.time()
        #print(len(docs))
        context = ""
        # Retrieved context from top k mentioned in the retrieval function 
        for k in range(len(docs)):
         context += docs[k][0].page_content
        print(len(context))
        llm_time=time.time()
        
        #Prediction from LLM QA chain through question and user query.
        res = llm_context_chain.predict(question = query,context = context)
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(res)
        torch.cuda.empty_cache()
        s2=time.time()
        return {"time to load model":round(endmodel-startmodel,2),"answer": res,"time of request":round((s2-s1)-(end-start),2),"time to answer":round((end-start),2),"LLM took time":round((end-llm_time),2),"time to retrieve data":round((time_ret-start),2)}
  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))
  




# # %%
