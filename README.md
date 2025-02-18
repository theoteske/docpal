# DocPal: Interactive Document Chat Interface

DocPal is a Streamlit-based application that enables interactive conversations with documents using Retrieval-Augmented Generation (RAG) and semantic search. Upload your documents and engage in natural language conversations with an LLM about their content.

## Features

- **Multiple Document Format Support**: Process PDF, DOCX, TXT, CSV, PPTX, and XLSX files
- **Semantic Search**: Utilizes advanced embedding models for context-aware document comprehension
- **Interactive Chat Interface**: Natural conversation flow with document context
- **Multi-Query Synthesis**: Generates multiple related queries to each user query and synthesizes their content for comprehensive answers
- **GPU Acceleration**: Automatic GPU utilization when available for improved performance

## How It Works

DocPal employs a multi-step process to generate comprehensive answers:

1. **Query Expansion**
   - When a user asks a question, the model generates multiple related queries
   - These queries explore different aspects of the original question
   - All generated queries are displayed to the user, providing transparency into the model's thinking process

2. **Multi-Perspective Analysis**
   - Each generated query (including the original) is processed independently
   - The model searches the document and generates specific answers for each query
   - All intermediate answers are shown to the user, allowing them to see how the final answer is constructed

3. **Answer Synthesis**
   - Uses Reciprocal Rank Fusion (RRF) to score and combine all generated answers
   - Weighs the relevance and importance of each response
   - Generates a final, comprehensive answer that incorporates the most relevant insights
   
4. **Conversation Management**
   - Maintains a persistent chat history across multiple user queries
   - Records both the final synthesized answers and the user's questions
   - Allows users to track the conversation flow and refer back to previous interactions

This approach enables DocPal to provide more nuanced and complete answers by considering multiple perspectives and making its reasoning process visible to the user.

## Architecture

- Frontend: Streamlit
- Embedding Model: Instructor XL (HuggingFace)
- Vector Store: ChromaDB
- Language Model: Ollama (llama3)
- Document Processing: LangChain document loaders

## Installation

1. Clone the repository:
```bash
git clone https://github.com/theoteske/docpal.git
cd docpal
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download and set up Ollama locally:
- Follow instructions at [Ollama's official website](https://ollama.ai/)
- Pull the llama3 model:
```bash
ollama pull llama3
```

## Usage

1. Start the application:
```bash
streamlit run main.py
```

2. Open your browser and navigate to the displayed localhost URL (typically http://localhost:8501)

3. Upload a supported document using the file upload interface

4. Start chatting with your document!

## Project Structure

```
docpal/
├── src/
│   └── config.py      # Configuration and file loader mappings
│   └── message.py        # Message class definitions
│   └── chat_service.py   # Core chat functionality
│   └── embeddings.py     # Embedding model utilities
│   └── main.py              # Application entry point
│   └── ui/
│       └──file_upload.py    # File upload interface
│       └── chat_interface.py # Chat interface
└── requirements.txt     # Project dependencies
```

## Future Enhancements

- [ ] Add support for more document formats
- [ ] Improve error handling and user feedback
- [ ] Add unit and integration tests
- [ ] Containerize the application using Docker
- [ ] Deploy to cloud platforms (AWS, GCP, Azure)
- [ ] Optimize for large-scale document processing

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [LangChain](https://www.langchain.com/)
- Using [Ollama](https://ollama.ai/) for LLM integration