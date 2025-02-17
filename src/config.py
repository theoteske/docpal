import logging
from langchain_community.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)

logger = logging.getLogger(__name__)

FILE_LOADERS = {
    'csv': CSVLoader,
    'docx': Docx2txtLoader,
    'pdf': PyMuPDFLoader,
    'pptx': UnstructuredPowerPointLoader,
    'txt': TextLoader,
    'xlsx': UnstructuredExcelLoader,
}

ACCEPTED_FILE_TYPES = list(FILE_LOADERS)