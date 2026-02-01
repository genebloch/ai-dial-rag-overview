import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY


SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION: 
{query}"""


class MicrowaveRAG:

    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self) -> VectorStore:
        """Initialize the RAG system"""
        print("🔄 Initializing Microwave Manual RAG System...")
        index_dir = "microwave_faiss_index"
        if os.path.isdir(index_dir):
            print("📦 Found existing FAISS index. Loading...")
            vectorstore = FAISS.load_local(
                folder_path=index_dir,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            print("📦 No existing index found. Creating a new one...")
            vectorstore = self._create_new_index()
        print("✅ Vector store ready")
        return vectorstore

    def _create_new_index(self) -> VectorStore:
        print("📖 Loading text document...")
        loader = TextLoader(file_path="task/microwave_manual.txt", encoding="utf-8")
        documents = loader.load()

        print("✂️ Splitting document into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", "."],
        )
        chunks = splitter.split_documents(documents)

        print("🧮 Creating FAISS vector store from chunks...")
        vectorstore = FAISS.from_documents(chunks, self.embeddings)

        print("💾 Saving index locally...")
        vectorstore.save_local("microwave_faiss_index")

        return vectorstore

    def retrieve_context(self, query: str, k: int = 4, score=0.3) -> str:
        """
        Retrieve the context for a given query.
        Args:
              query (str): The query to retrieve the context for.
              k (int): The number of relevant documents(chunks) to retrieve.
              score (float): The similarity score between documents and query. Range 0.0 to 1.0.
        """
        print(f"{'=' * 100}\n🔍 STEP 1: RETRIEVAL\n{'-' * 100}")
        print(f"Query: '{query}'")
        print(f"Searching for top {k} most relevant chunks with similarity score {score}:")

        results = self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            score_threshold=score,
        )

        context_parts = []
        for item in results:
            # `item` can be a tuple (Document, score) or an object with `.document` & `.score`
            try:
                doc, doc_score = item
            except Exception:
                doc = getattr(item, "document", None) or item[0]
                doc_score = getattr(item, "score", None) if hasattr(item, "score") else item[1]

            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            context_parts.append(content)
            print(f"➡️  Score: {doc_score:.4f}")
            print(f"📄 Chunk:\n{content}\n")

        print("=" * 100)
        return "\n\n".join(context_parts) # will join all chunks ion one string with `\n\n` separator between chunks

    def augment_prompt(self, query: str, context: str) -> str:
        print(f"\n🔗 STEP 2: AUGMENTATION\n{'-' * 100}")

        augmented_prompt = USER_PROMPT.format(context=context, query=query)

        print(f"{augmented_prompt}\n{'=' * 100}")
        return augmented_prompt

    def generate_answer(self, augmented_prompt: str) -> str:
        print(f"\n🤖 STEP 3: GENERATION\n{'-' * 100}")
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt),
        ]

        response = self.llm_client.invoke(messages)
        print(f"🧠 Response:\n{response.content}\n{'=' * 100}")
        return response.content


def main(rag: MicrowaveRAG):
    print("🎯 Microwave RAG Assistant")

    while True:
        user_question = input("\n> ").strip()
        if not user_question:
            continue

        # Step 1: Retrieval
        context = rag.retrieve_context(user_question)
        # Step 2: Augmentation
        augmented = rag.augment_prompt(user_question, context)
        # Step 3: Generation
        answer = rag.generate_answer(augmented)

        print(f"\n💬 Answer:\n{answer}")



main(
    MicrowaveRAG(
        embeddings=AzureOpenAIEmbeddings(
            deployment="text-embedding-3-small-1",
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY),
        ),
        llm_client=AzureChatOpenAI(
            temperature=0.0,
            azure_deployment="gpt-4o",
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY),
            api_version="",
        ),
    )
)