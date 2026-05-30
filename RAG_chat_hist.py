# imports
from chunk_vector_store import ChunkVectorStore as cvs
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# imports for integrating chat history
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


# import streamlit for session_id management to keep track of chat history for each user session
import streamlit as st

# import ChatGroq for using Groq LLM instead of Ollama and load the api key from .env file
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

class Rag: # create a blueprint for an object

    vector_store=None
    retriever=None
    chain=None
    # init constructor
    def __init__(self):
        self.cvs_obj=cvs() # for vector store
        # only prompt is not neede here
        # self.prompt=PromptTemplate(
        #     template='Answer the user question \nUser Question: {question}\nIf you do not know the answer, say "I do not know."',
        #     input_variables=['question']
        # )

        # for creating history aware retriever
        # use contextualize query prompt to reformulate the user's latest question based on chat history 
        self.contextualize_q_system_prompt="""Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        self.contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ('system',self.contextualize_q_system_prompt),
                MessagesPlaceholder('chat_history'),
                ('human','{input}')
            ]
        )

        # for creating qa chain
        self.qa_system_prompt="""You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Use three sentences maximum and keep the answer concise.\

        {context}"""
        self.qa_prompt=ChatPromptTemplate.from_messages(
            [
                ('system',self.qa_system_prompt),
                MessagesPlaceholder('chat_history'),
                ('human','{input}')
            ]
        )
        # replacing ChatOllama with ChatGroq
        # self.model=ChatOllama(model='qwen3:0.6b')
        self.model=ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.7
        )
        # store dictionary to keep track of chat history
        self.store={}
    # retrieval step
    def set_retriever(self):
        self.retriever=self.vector_store.as_retriever(search_type='mmr',search_kwargs={'k':3,'lambda_mult':0.5})
        # also create history aware retriever
        self.history_aware_retriever=create_history_aware_retriever(self.model,self.retriever,self.contextualize_q_prompt)
    # augmentation step
    def augment(self):
        # self.chain=({'context':self.retriever,'question':RunnablePassthrough()},
        #             self.prompt|self.model|StrOutputParser()
        #             )

        # create a chain which consists of history aware retriever and qa chain
        # create qa chain
        self.qa_chain=create_stuff_documents_chain(self.model,self.qa_prompt)
        # rag chain
        self.rag_chain=create_retrieval_chain(self.history_aware_retriever,self.qa_chain)
        # final chain
        self.chain=RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
            input_messages_key='input',
            history_messages_key='chat_history',
            output_messages_key='answer'
        )

    # generation
    def ask(self,query):
        # check if chai is not empty
        if self.chain == None:
            return 'Please upload a PDf for context'
        return self.chain.invoke({'input':query},{'configurable':{'session_id':st.session_state['session_id']}})['answer']

    # feed pdf i.e. store pdf file in vector db
    def feed(self,file_path):
        # get chunks
        chunks=self.cvs_obj.split_into_chunks(file_path)
        # store in vector db variable of object
        self.vector_store=self.cvs_obj.store_to_vector_database(chunks)

        # set retriever because now we have split and embedded the uploaded doc and stored in vector db
        self.set_retriever()
        # define the chain for augmentation step
        self.augment()

    # clear memory
    def clear(self):
        self.vector_store=None
        self.retriever=None
        self.chain=None

    # function to get_session_history 
    def get_session_history(self,session_id:str)->BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id]=ChatMessageHistory()
        return self.store[session_id]
