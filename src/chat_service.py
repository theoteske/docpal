import json
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
import streamlit as st

from message import HumanMessage, AIMessage
from embeddings import load_model
from config import FILE_LOADERS, logger

class ChatWithFile:
    """Manages chat interactions with documents using LLM and vector storage.

    This class handles document processing, embedding, storage, and maintains
    conversation history.

    Attributes:
        embedding_model: Model used for text embeddings.
        vectordb: Vector database for storing document embeddings.
        memory: Buffer for maintaining chat history.
        llm: Language model instance.
        qa: Conversational retrieval chain.
        conversation_history: List of chat messages.
    """
    def __init__(self, file_path: str, file_type: str):
        """Initialize chat service."""
        self.embedding_model = load_model()
        self.vectordb = None
        self.initialize_chat(file_path, file_type)

    def initialize_chat(self, file_path: str, file_type: str):
        """Set up chat components including document processing and LLM chain.

        Args:
            file_path: Path to the uploaded document.
            file_type: Extension of the uploaded file.
        """
        loader = FILE_LOADERS[file_type](file_path=file_path)
        pages = loader.load_and_split()
        docs = self.split_into_chunks(pages)
        self.store_in_chroma(docs)

        self.memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

        self.llm = Ollama(model='llama3')

        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            self.vectordb.as_retriever(search_kwargs={'k': 10}),
            memory=self.memory
        )

        self.conversation_history = []

    def split_into_chunks(self, pages: list) -> list:
        """Split document pages into semantic chunks for processing.

        Args:
            pages: List of document pages to be split.

        Returns:
            List of semantic chunks ready for embedding.
        """
        text_splitter = SemanticChunker(
            embeddings=self.embedding_model,
            breakpoint_threshold_type="percentile"
        )
        return text_splitter.split_documents(pages)

    def simplify_metadata(self, doc):
        """Convert complex metadata values to strings for storage.

        Args:
            doc: Document with metadata to be simplified.

        Returns:
            Document with simplified metadata.
        """
        metadata = getattr(doc, "metadata", None)
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if isinstance(value, (list, dict)):
                    metadata[key] = str(value)
        return doc

    def store_in_chroma(self, docs: list):
        """Store document chunks in Chroma vector database.

        Args:
            docs: List of document chunks to be stored.
        """
        docs = [self.simplify_metadata(doc) for doc in docs]
        self.vectordb = Chroma.from_documents(docs, embedding=self.embedding_model)
        self.vectordb.persist()

    def reciprocal_rank_fusion(self, all_results: list) -> list:
        """Combine and rank query results using reciprocal rank fusion.

        Args:
            all_results: List of query results to be ranked.

        Returns:
            List of results sorted by fusion score.
        """
        fused_scores = {}
        for result in all_results:
            doc_id = result["query"]
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"doc": result, "score": 0}
            fused_scores[doc_id]["score"] += 1

        reranked_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        return reranked_results

    def create_synthesis_prompt(self, original_question: str, all_results: list) -> str:
        """Create a prompt for synthesizing multiple search results.

        Args:
            original_question: The user's initial question.
            all_results: Ranked list of search results.

        Returns:
            Formatted prompt string for synthesis.
        """
        sorted_results = sorted(all_results, key=lambda x: x["score"], reverse=True)
        prompt = (
            f"Based on the user's original question: '{original_question}', "
            "here are the answers to the original and related questions, "
            "ordered by their relevance (with RRF scores). Please synthesize "
            "a comprehensive answer focusing on answering the original "
            "question using all the information provided below, ensuring "
            "that the answer is not overly verbose and is relevant to the "
            "original question:\n\n"
        )

        for idx, result in enumerate(sorted_results):
            prompt += f"Answer {idx + 1} (Score: {result['score']}): {result['answer']}\n\n"

        prompt += (
            "Given the above answers, especially considering those with "
            "higher scores, please provide the best possible composite answer "
            "to the user's original question."
        )

        return prompt

    def extract_json_from_response(self, response_text: str) -> tuple:
        """Extract JSON array from LLM response text.

        Args:
            response_text: Raw text response from LLM.

        Returns:
            Tuple containing parsed JSON data or empty tuple if parsing fails.
        """
        json_result = ()
        try:
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            json_str = response_text[json_start:json_end]
            json_result = json.loads(json_str)
        except (ValueError, json.JSONDecodeError) as e:
            logger.error("Failed to parse JSON: %s", e)
        return json_result

    def generate_related_queries(self, original_query: str) -> tuple:
        """Generate related search queries based on original question.

        Args:
            original_query: User's initial question.

        Returns:
            Tuple of related queries or empty tuple if generation fails.
        """
        prompt = (
            f"In light of the original inquiry: '{original_query}', let's "
            "delve deeper and broaden our exploration. Please construct a "
            "JSON array containing four distinct but interconnected search "
            "queries. Each query should reinterpret the original prompt's "
            "essence, introducing new dimensions or perspectives to "
            "investigate. Aim for a blend of complexity and specificity in "
            "your rephrasings, ensuring each query unveils different facets "
            "of the original question. This approach is intended to "
            "encapsulate a more comprehensive understanding and generate the "
            "most insightful answers possible. Only respond with the JSON "
            "array itself."
        )
        response = self.llm.invoke(input=prompt)

        if hasattr(response, 'content'):
            generated_text = response.content
        elif isinstance(response, dict):
            generated_text = response.get('content')
        else:
            generated_text = str(response)

        related_queries = self.extract_json_from_response(generated_text)
        return related_queries

    def chat(self, question: str) -> dict:
        """Process user question and generate comprehensive response.

        Args:
            question: User's question about the document.

        Returns:
            Dict containing answer and any additional response information.
        """
        related_queries_dicts = self.generate_related_queries(question)
        related_queries_list = [q["query"] for q in related_queries_dicts]
        queries = [question] + related_queries_list

        all_results = []

        for query_text in queries:
            response = self.qa.invoke(query_text)
            if response:
                st.write("Query: ", query_text)
                st.write("Response: ", response["answer"])
                all_results.append(
                    {
                        "query": query_text,
                        "answer": response["answer"]
                    }
                )
            else:
                st.write("No response received for: ", query_text)

        if all_results:
            reranked_results = self.reciprocal_rank_fusion(all_results)
            scored_results = [{"score": res["score"], **res["doc"]} for res in reranked_results]
            synthesis_prompt = self.create_synthesis_prompt(question, scored_results)
            synthesized_response = self.llm.invoke(synthesis_prompt)

            if synthesized_response:
                st.write(synthesized_response)
                final_answer = synthesized_response
            else:
                final_answer = "Unable to synthesize a response."

            self.conversation_history.append(HumanMessage(content=question))
            self.conversation_history.append(AIMessage(content=final_answer))

            return {"answer": final_answer}

        self.conversation_history.append(HumanMessage(content=question))
        self.conversation_history.append(AIMessage(content="No answer available."))
        return {"answer": "No results were available to synthesize a response."}