import os
import streamlit as st
from src.chat_service import ChatWithFile

def chat_interface():
    st.title('DocPal - Talk to a Document')
    file_path = st.session_state.get('file_path')
    file_type = st.session_state.get('file_type')
    if not file_path or not os.path.exists(file_path):
        st.error('File missing. Please go back and upload a file.')
        return

    if 'chat_instance' not in st.session_state:
        st.session_state['chat_instance'] = ChatWithFile(
            file_path=file_path,
            file_type=file_type
        )

    user_input = st.text_input('Ask a question about your document:')
    if user_input and st.button('Send'):
        with st.spinner('Thinking...'):
            top_result = st.session_state['chat_instance'].chat(user_input)

            if top_result:
                st.markdown('**Top Answer:**')
                st.markdown(f"> {top_result['answer']}")
            else:
                st.write('No top result available.')

            st.markdown('**Chat History:**')
            for message in st.session_state['chat_instance'].conversation_history:
                prefix = '*You:* ' if message.__class__.__name__ == 'HumanMessage' else '*AI:* '
                st.markdown(f'{prefix}{message.content}')