import streamlit as st
import torch
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from InstructorEmbedding import INSTRUCTOR

@st.cache_resource
def load_model():
    """Load and initialize the embedding model with appropriate device settings.

    Returns:
        HuggingFaceInstructEmbeddings: Initialized embedding model.

    Raises:
        Exception: If model loading fails, displays error in Streamlit UI and stops execution.
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        st.warning('CUDA is not available. Falling back to CPU. This may result in slower performance.')

    with st.spinner(f'Downloading Instructor XL Embeddings Model locally on {device}...please be patient'):
        try:
            embedding_model = HuggingFaceInstructEmbeddings(
                model_name="hkunlp/instructor-large",
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': False},
                client=INSTRUCTOR("hkunlp/instructor-large", cache_folder=None, device=device)
            )
            st.success('Model loaded successfully.')
        except Exception as e:
            st.error(f'Failed to load the embedding model: {e}')
            st.stop()

    return embedding_model