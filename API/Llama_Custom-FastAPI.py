# %%
#%pip install "transformers" "accelerate" "langchain" "einops"
#%pip install chromadb
#%pip install langchain
#%pip install sentence_transformers


# %%
import time
import os
import torch
from transformers import pipeline
from transformers import AutoTokenizer , AutoModelForCausalLM
from fastapi import FastAPI, HTTPException
app = FastAPI()
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
persist_directory = "Databases/500_DB_GTR" # add path to specific vector database
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=persist_directory,
        anonymized_telemetry=False
)

# %%
# Vector Database instance
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)

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
<<SYS>>:You are an Intelligent and helpful Chatbot specifically designed to answer user's question/commands on the basis of the context given by the user.
You hate being wrong and always prefer being factually correct and don't mind the time taken.You value the User's time and always answer within 150 words.You are always careful and recheck your answers to ensure they are complete. You are very humble and always refuse to answer if the question does not match the context.<</SYS>>
[INST]
USER:Answer the Question:"{question}" based on the Context:"{context}".
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

        # Get the answer from the chain
        start = time.time()
        
        
        # similarity search from vectorDB to get context
        docs = db.similarity_search_with_score(query,k=2)
        #print(len(docs))
        context = ''
        # Retrieved context. 
        for i in range(len(docs)):
         context += ' '+docs[i][0].page_content
        print(len(context))
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
        return {"time to load model":round(endmodel-startmodel,2),"answer": res,"time of request":round((s2-s1)-(end-start),2),"time to answer":round((end-start),2),"context sent to model":context}
  except Exception as e:
      raise HTTPException(status_code=500, detail=str(e))
  



