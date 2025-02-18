import streamlit as st
from ui.file_upload import upload_and_handle_file
from ui.chat_interface import chat_interface

def main():
    if 'page' not in st.session_state:
        st.session_state['page'] = 1

    if st.session_state['page'] == 1:
        upload_and_handle_file()
    elif st.session_state['page'] == 2:
        chat_interface()

if __name__ == '__main__':
    main()