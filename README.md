# Advanced Multi-LLM Agentic RAG Pipeline

A robust, self-correcting Retrieval-Augmented Generation (RAG) system built with [LangGraph](https://python.langchain.com/docs/langgraph). This pipeline intelligently decides when to retrieve documents, filters for relevance, checks for context-grounding (hallucination prevention), and rewrites queries if the generated answer isn't useful—all while orchestrating multiple LLM providers (Cerebras, Groq, and Gemini) for optimal performance and cost.

## 🚀 Key Features

* **Intelligent Routing:** Evaluates incoming questions to decide if internal document retrieval is necessary or if the model can answer directly from general knowledge.
* **Multi-LLM Orchestration:** Leverages different models for different tasks based on their strengths:
    * **Groq:** Fast structured JSON outputs for decision-making (Retrieval, Relevance, Usefulness, Query Rewriting).
    * **Cerebras:** Heavy lifting for direct generation and context-based answering.
    * **Gemini:** Strict revision and formatting of unsupported answers.
* **Self-Correction & Hallucination Prevention:** * **Grounding Check (`is_sup`):** Validates that the generated answer is strictly supported by the retrieved context.
    * **Revision Loop:** If an answer includes unsupported claims, it loops back to a strict reviser to strip out ungrounded information.
* **Query Rewriting:** If the final answer is deemed "not useful" to the user's prompt, the system automatically rewrites the query for better vector retrieval and tries again.
* **Vector Search:** Local Qdrant instance paired with `BAAI/bge-large-en-v1.5` embeddings.

## 🧠 Architecture Flow

The pipeline follows a stateful graph architecture:

1.  **`decide_retrieval`**: Determines if the query needs external data.
2.  **`retrieve`**: Fetches top-k chunks from Qdrant.
3.  **`is_relevant`**: Filters out noisy/irrelevant chunks.
4.  **`generate_from_context`**: Drafts an initial answer.
5.  **`is_sup`**: Evaluates if the context *fully supports* the drafted answer.
6.  **`revise_answer`**: Strips out unsupported claims (loops until supported).
7.  **`is_useful`**: Evaluates if the grounded answer actually solves the user's query.
8.  **`rewrite_question`**: Adjusts the search query if the answer wasn't useful (loops back to retrieval).

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Framework** | LangChain, LangGraph |
| **LLMs** | Cerebras (`gpt-oss-120b`), Groq (`gpt-oss-120b`), Google Gemini (`gemini-2.5-flash-lite`) |
| **Embeddings** | HuggingFace (`BAAI/bge-large-en-v1.5`) |
| **Vector Database**| Qdrant (Local) |
| **Data Models** | Pydantic, Python `TypedDict` |

## 📦 Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/agentic-rag-pipeline.git](https://github.com/yourusername/agentic-rag-pipeline.git)
cd agentic-rag-pipeline
```

**2. Install dependencies**
```bash
pip install langchain langchain-google-genai langchain-cerebras langchain-groq langchain-huggingface langgraph qdrant-client langchain-qdrant pydantic python-dotenv pypdf
```

**3. Environment Variables**
Create a `.env` file in the root directory and add your API keys:
```env
CEREBRAS_API_KEY=your_cerebras_key
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_gemini_key
```

**4. Prepare your Data**
Place your target PDFs (e.g., `Company_Policies.pdf`, `Company_Profile.pdf`, `Product_and_Pricing.pdf`) in a `/content/` directory (or update the paths in the script to match your local structure).

## 💻 Usage

Run the script to initialize the vector database, compile the graph, and invoke a query:

```python
# The pipeline is compiled into a LangGraph app
result = app.invoke({
    'query': 'who is CEO of NexaAI',
    'docs': [],
    'answer': "",
    'relevant_docs': [],
    'context': ''
})

# Access pipeline metadata and output
print("Answer:", result.get("answer"))
print("Supported by context:", result.get("is_support"))
print("Evidence:", result.get("evidence"))
print("Retries needed:", result.get("retries", 0))
```

## 📂 Project Structure

* `main.py` - Contains the full LangGraph node definitions, edges, state models, and execution logic.
* `company_db/` - (Generated) Local directory storing the Qdrant vector database.
* `.env` - API keys for LLM providers.
