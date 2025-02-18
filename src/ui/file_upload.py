import os
import pathlib
import streamlit as st
from src.config import ACCEPTED_FILE_TYPES

def upload_and_handle_file():
    st.title("DocPal - Talk to a Document")
    uploaded_file = st.file_uploader(
        label=(
            f"Choose a {', '.join(ACCEPTED_FILE_TYPES[:-1]).upper()}, or "
            f"{ACCEPTED_FILE_TYPES[-1].upper()} file"
        ),
        type=ACCEPTED_FILE_TYPES
    )

    if uploaded_file:
        file_type = pathlib.Path(uploaded_file.name).suffix
        file_type = file_type.replace(".", "")

        if file_type:
            csv_pdf_txt_path = os.path.join("temp", uploaded_file.name)
            if not os.path.exists("temp"):
                os.makedirs("temp")
            with open(csv_pdf_txt_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.session_state["file_path"] = csv_pdf_txt_path
            st.session_state["file_type"] = file_type
            st.success(f"{file_type.upper()} file uploaded successfully.")
            st.button(
                "Proceed to Chat",
                on_click=lambda: st.session_state.update({"page": 2})
            )
        else:
            st.error(
                f"Unsupported file type. Please upload a "
                f"{', '.join(ACCEPTED_FILE_TYPES[:-1]).upper()}, or "
                f"{ACCEPTED_FILE_TYPES[-1].upper()} file."
            )