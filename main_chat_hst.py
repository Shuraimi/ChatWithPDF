# used to write the streamlit code to deploy the app
import streamlit as st
import tempfile
import os

# import Rag class
# from RAG_projects.ChatWithPDF_with_chathistory.RAG_chat_hist import Rag
from RAG_chat_hist import Rag

#display messages stored in session_state
def display_messages():
    for message in st.session_state['messages']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

def process_file():
    """
    This function handles:
    clearing old chat
    saving uploaded files temporarily
    feeding them into RAG
    deleting temporary files
    """
    
    #clearing old chat, chat history and vector db retriver and chain
    st.session_state['assistant'].clear()
    st.session_state['messages']=[]
    
    # loop thru uploaded files
    for file in st.session_state['file_uploader']:
        # file here is an UploadedFile object
        # therefore proceed with temporary file creation because doc loaders require a file path and this tempfile creation creates a temporary file in disk 
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            # write the contents of uploaded file to the temp file object
            tf.write(file.getbuffer()) # This writes uploaded PDF bytes into temp file
            # get buffer gets raw binary data from uploaded file
            
            # save path
            file_path=tf.name
            
        # spinner ui to show loading animation
        # and feed file to vector storage
        with st.session_state["feeder_spinner"], st.spinner("Uploading the file"):
            # feed into RAG
            st.session_state['assistant'].feed(file_path)
            # this is where actual RAG pipeline starts
        # delete tempfile
        os.remove(file_path)

def process_input():
    """
    Is used to take in prompt from user(using the chat_input component), display it using chat_message component, get assistant's response and also display it using chat_message component
    """
    
    if prompt :=st.chat_input('What do you want to ask?'):
        # display the user's input 
        with st.chat_message('user'):
            st.markdown(prompt)
        # append it to the chat history
        st.session_state.messages.append({'role':'user','content':prompt})
        
        # get assistant's response
        # where assistant is a RAG object
        response=st.session_state['assistant'].ask(prompt)
        with st.chat_message('assistant'):
            st.markdown(response)
        #append response back to the chat history
        st.session_state.messages.append({'role':'assistant','content':response})
        
def main():
    st.title('RAG app')
    st.image('App flow.png',width=300,caption='RAG app architecture',)
    
    # to get unique sesion id for each user to keep track of previous chat intercations of user and use that as context for future interactions, we can use uuid module to generate unique session ids
    import uuid
    if 'session_id' not in st.session_state:
        st.session_state['session_id']=str(uuid.uuid4())
    st.write(f"Your session id is {st.session_state['session_id']}") # this is just for testing, in production you would not want to display session id to user
    
    # initalise session_state
    if 'assistant' not in st.session_state:
        st.session_state['assistant']=Rag()
        st.session_state['messages']=[]
    # streamlit upload gives a FILE OBJECT, not a normal file path
    # but RAG app expects file path. So it creates a temporary physical file using tempfile

    # for uploading file
    st.file_uploader('Upload the document',
                    type=['pdf'],
                    key='file_uploader', # so the files get stored in st.session_state['file_uploader]
                    on_change=process_file, #An optional callback invoked when this file_uploader's value changes
                    label_visibility='collapsed',
                    accept_multiple_files=True
                    )
    st.session_state['feeder_spinner']=st.empty()
    display_messages()
    process_input()
if __name__=='__main__':
    main()
    
    
        