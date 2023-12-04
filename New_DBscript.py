# Add ability to use only one GPU for embeddings.
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import FAISS , Chroma
import faiss
from pathlib import Path
from langchain.embeddings import HuggingFaceBgeEmbeddings
import time
import os , sys
#Setting up Model
model_name = "BAAI/bge-large-en-v1.5"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
embeddings = HuggingFaceBgeEmbeddings(
model_name=model_name,
encode_kwargs=encode_kwargs)

#Setting up for API call
if len(sys.argv) < 2:
 print("Usage: python my_script.py <variable>")
 sys.exit(1)
# Get the variable passed as an argument (the first argument, sys.argv[1])
link = sys.argv[1]
link = link.replace('https://','')
link = link.replace('/','')

path = "VDatabases/"+link

#Loading up all documents inside a folder - Gives error if error occured.
try:
    print("-> Loading Documents from the folder\n")
    #Specify Directory to load docu from
    docu_directory = 'CrawledData/'+link
    #Loads documents
    loader = DirectoryLoader(docu_directory,glob="**/*.txt",show_progress=True)
    docs = loader.load()
    print('\n',len(docs),"Documents Loaded")

    # Splitting the documents into chunks.
    print("\n->  Splitting Documents into chunks")
    chunk_size = 300
    overlap_size = 10
    splitter = TokenTextSplitter(chunk_size=chunk_size,chunk_overlap = overlap_size)
    texts = splitter.split_documents(docs)
    print("\n Split texts into chunks of",chunk_size,"and overlap",overlap_size)

    #Check if the folder VDatabases exists
    
    if not os.path.isdir('VDatabases'):
       os.system("mkdir VDatabases")
    
    # Making embeddings from the documents
    print("\n-> Creating Embeddings")
    start = time.time()
    db = FAISS.from_documents(texts,embeddings)
    #Saving Embeddings as a file
    db.save_local(path)
    end = time.time()
    print("\n-> Finished Embedding and saved to ",path,", Time taken to embed:",end-start," seconds")

    """
    Writing path to env , will be used to dynamically talk with different chatbots.
    
    """
    with open(".env","a+") as f: 
       f.write('\n'+f'{link}=VDatabases/{link}')
    f.close()
    print(f"\n Ingestion complete and site added to Database! \n\n\n\n\n")

#Catching errors
except Exception as error:
    print("Error Occured->", error)


