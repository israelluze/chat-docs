import os
import streamlit as st
import tempfile

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from decouple import config


os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')
persist_directory = 'db'


class Process():
    
    

    @staticmethod
    def process_pdf(file):
        #persist file in a directoty
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        os.remove(temp_file_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
        )

        chunks = text_splitter.split_documents(documents=docs)
        return chunks
    
    @staticmethod
    def load_existing_vector_store():
        if os.path.exists(os.path.join(persist_directory)):
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=OpenAIEmbeddings()
            )
            return vector_store
        return None
    
    @staticmethod
    def add_to_vector_store(chunks, vector_store=None):
        if vector_store:
            vector_store.add_documents(chunks)
        else:
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=OpenAIEmbeddings(),
                persist_directory=persist_directory
            )
        return vector_store
    
    @staticmethod
    def ask_question(model, query, vector_store: Chroma):
        llm = ChatOpenAI(model=model)
        retriever = vector_store.as_retriever()

        system_prompt = '''
        Use o contexto para responder as perguntas.
        Se não encontrar a resposta no contexto, explique não há informações disponíveis.
        Responsa em formato markdowne e com visualizações elaboradas e interativas.
        Contexto: {context}
        '''

        messages = [('system', system_prompt)]
        for message in st.session_state.messages:
            messages.append((message.get('role'), message.get('content')))
        messages.append(('human', '{input}'))

        prompt = ChatPromptTemplate.from_messages(messages)

        question_answer_chain = create_stuff_documents_chain(
            llm=llm, 
            prompt=prompt,
        )

        chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=question_answer_chain
        )

        response = chain.invoke({'input': query})
        return response.get('answer')

