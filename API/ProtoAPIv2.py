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
# from spellchecker import SpellChecker
import json

# def correct_query(input_sentence):
#     spell = SpellChecker()
    
#     # Tokenize the input sentence into words:
#     words = input_sentence.split()

#     # Create a list to store the corrected words
#     corrected_words = []

#     for word in words:
#         # Check if the word is misspelled
#         if spell.unknown([word]):
#             # Get the most likely correct spelling
#             corrected_word = spell.correction(word)

#             if corrected_word:
#                 corrected_words.append(corrected_word) 
#             else:
#                 corrected_words.append(word)
 
#         else:
#             # If the word is already correct, add it to the list as is
#             corrected_words.append(word)


#     # Reconstruct the corrected sentence
#     corrected_sentence = ' '.join(corrected_words)
    
#     return corrected_sentence

from transformers import pipeline
from transformers import AutoTokenizer , AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from langchain.embeddings import HuggingFaceBgeEmbeddings
#loading dotenv file containing path links to databases , this eliminates the scope problem to manipulate env
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
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def main():
    return {"message": "Hello World"}
@app.post("/db_script",response_model=str)
async def run_db_script(link:str, chatbot_id:str):
   url = link
   id = chatbot_id
   os.system("python New_DBscript.py "+url+" "+id)
   string = 'DBscript run successfully'
   return 'DBscript run successfully'
  
  

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
model = AutoModelForCausalLM.from_pretrained("Llama-2-7b-chat-hf",device_map="auto",torch_dtype=torch.float16)
endmodel = time.time()
print(f"Loading model took: {round(endmodel-startmodel,2)} s")


# load model pipeline
generate_text = pipeline("text-generation",model=model, tokenizer = tokenizer,
                         trust_remote_code=True, device_map= "auto", 
                         do_sample=True,eos_token_id = tokenizer.eos_token_id,
                         max_new_tokens = 512,)


# %%
# Hugging Face model pipeline
from langchain.llms import HuggingFacePipeline

hf_pipeline = HuggingFacePipeline(pipeline=generate_text,model_kwargs={'temperature':0})

# %%
from langchain.vectorstores import Chroma,FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from chromadb.config import Settings

# %%
# Text Embedding Model
model_name = "BAAI/bge-large-en-v1.5"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs)

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
<<SYS>> You are a robot named "AI assistant".The robot has a slot where I(Your master) insert a memory which contains various documents.The documents are then transferred in your knowledge base. 
<</SYS>>
Documents in your knowledge base:{info} \n\n
[/INST]
[INST] 
<<SYS>>Now , face the user and answer his question based on the documents in your knowledge base without mentioning the question.
Your response should not exceed 100 words.
<</SYS>>
User's question:'{question}'
[/INST]
'''
# <<SYS>>
# Answer only based on the context.
# Respond with "Hmm , I don't know" if the context cannot answer the question.
# <</SYS>>
# prompt chain
prompt_with_context = PromptTemplate(
input_variables=["question", "info"],
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

@app.post('/api/setpath',response_model=dict)
async def setpath(path_number:str):
    load_dotenv()
    path_number = path_number.replace('https://','')
    path_number = path_number.replace('/','')
    path = os.getenv(path_number)
    
    os.environ["DB_Path"] = path
    return {"path set to":os.getenv("DB_Path")}
    print(persist_directory)
   
   
@app.post('/api/askquery', response_model=dict)
async def askquery(question: str):
  
  
  s1=time.time()
  try:
     while True:
        load_dotenv() #This eliminates the need to restart server everytime a new db is added.
        #Whenever bot starts , cuda set to 1(Either set own value) , comment out the next line for parallel processing
        # os.system('export CUDA_VISIBLE_DEVICES=1')
        # Get the user query 
        if os.getenv("DB_Path") == "default":
            return {"answer":"Please select a valid chatbot to interact."}
        query = question.lower()
        # if query == "exit":
        #     break
        if query.strip() == "":
            continue

        #Chroma Settings initialised everytime the site is changed
        persist_directory=os.getenv("DB_Path")

        # Get the answer from the chain
        start = time.time()
        db = FAISS.load_local(persist_directory,embeddings)
        
        # question = correct_query(question)
        top_k=4
        # # similarity search from vectorDB to get context
        docs = db.similarity_search_with_score(query,k=top_k)
        # # MMR seach from vectorDB to get context
        # retriever = db.as_retriever(search_type="mmr",search_kwargs={"k":top_k,"lambda_mult":0.5})
        # docs = retriever.get_relevant_documents(query)
        #print(docs)
        time_ret= time.time()
        #print(len(docs))
        # Retrieved context from top k mentioned in the retrieval function 

        
        #For MMR
        # i=0
        # data = {}
        # for Document in docs:
        #  data[f"document_{i+1}"] = Document.page_content
        #  i+=1
        #Function
        def getLink(input_string):
            pattern = r'.*https'
            result_string = re.sub(pattern, 'https', input_string)
            result_string = result_string.replace('https', 'https://').replace('.com', '.com/').replace('.in', '.in/').replace('.txt', '')
            return result_string

        #For Similarity Search
        data = {}
        temp = 0
        for i in range(top_k):
            doc_key = f"doc_{i+1}"
            # temp+=len(docs[i][0].page_content)+ len( getLink(docs[i][0].metadata["source"]))
            # if(temp>12800): break
            if docs[i][1]<0.85:                  
             data[doc_key] = {
                "context" : docs[i][0].page_content,
                # "link" : getLink(docs[i][0].metadata["source"])
             }
        if len(data)==0:
           return{"answer":"Please ask a relevant question , or take the help of the sitemap for the same"} 
        else:
            context = json.dumps(data , indent = 4)
            
            print(len(context))
            llm_time=time.time()
            
            #Prediction from LLM QA chain through question and user query.
            res = llm_context_chain.predict(question = query,info = context)
            end = time.time()

            # Print the result
            print("\n\n> Question:")
            print(query)
            print(f"\n> Answer (took {round(end - start, 2)} s.):")
            print(res)
            s2=time.time()
            return {"question":question,"time to load model":round(endmodel-startmodel,2),"answer": res,"time of request":round((s2-s1)-(end-start),2),"time to answer":round((end-start),2),"LLM took time":round((end-llm_time),2),"time to retrieve data":round((time_ret-start),2),"context sent to model":context}
  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))
  




# # %%

# %%
