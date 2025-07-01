import os
import streamlit as st

from decouple import config

from process import Process

os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')


vector_store = Process.load_existing_vector_store()

st.set_page_config(
    page_title='Chat DOC',
    page_icon='🧾',
)
st.header('🤖 Chat with your documents')

with st.sidebar:
    st.header('Upload your documents')
    uploaded_files = st.file_uploader(
        label='Upload your PDF files',
        type=['pdf'],
        accept_multiple_files=True,
        help='You can upload multiple PDF files to chat with them.',
    )

    if uploaded_files:
        with st.spinner('Processing your files...'):
            
            all_chunks = []
            for uploaded_file in uploaded_files:
                chunks = Process.process_pdf(file=uploaded_file)
                all_chunks.extend(chunks)

            vector_store = Process.add_to_vector_store(
                chunks=all_chunks,
                vector_store=vector_store,
            )

    model_options = [
        'gpt-3.5-turbo',
        'gpt-4',
        'gpt-4-1106-preview',
        'gpt-4-vision-preview',
        'gpt-4o',
        'gpt-4o-mini',
    ]

    selected_model = st.selectbox(
        label='Select a model',
        options=model_options,
    )

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

question = st.chat_input('How can I help you?')

if vector_store and question:
    for message in st.session_state.messages:
        st.chat_message(message.get('role')).write(message.get('content'))

    st.chat_message('user').write(question)
    st.session_state.messages.append({'role':'user', 'content': question})

    response = Process.ask_question(
        model = selected_model,
        query = question,
        vector_store = vector_store,
    )

    st.chat_message('ai').write(response)
    st.session_state.messages.append({'role':'ai', 'content': response})


