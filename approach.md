# Technical Approach: SHL Assessment Recommender

## 1. Problem Understanding
The goal is to build a stateless conversational API (`POST /chat`) that acts as an SHL assessment recommender. The system must process an ongoing chat history, interpret vague intents, ask clarifying questions, refine assessment shortlists based on evolving constraints, and compare requested assessments. Importantly, the outputs must be strictly grounded in the provided `data/shl_catalog_main.json` catalog, preventing any hallucination of product names or URLs while adhering to a strict JSON schema.

## 2. System Architecture
The application is built using **FastAPI** for high-performance HTTP request handling, combined with **LangGraph** to orchestrate the internal conversational logic as a Directed Acyclic Graph (DAG) per request. 

Because the API must be completely stateless, we do not utilize LangGraph's persistent checkpointers or database state. Instead, for every request, the full conversation history is processed through the graph nodes:
1. **Validation & Guardrails**: Rejects off-topic, legal, or prompt-injection queries.
2. **Intent Detection**: Classifies the conversation into `clarify`, `recommend`, `refine`, `compare`, `refuse`, or `done`.
3. **Context Extraction**: Uses regex/keywords to pull roles, skills, and preferences from the text.
4. **Retrieval & Reranking**: Executes local vector/keyword search against the loaded catalog.
5. **LLM Reply Generation**: Calls Gemini 2.5 Flash purely for the human-readable text.
6. **Response Assembly**: Safely formats the strictly-typed JSON response.

## 3. Catalog & Data Setup
On application startup, the `app/catalog_loader.py` reads `data/shl_catalog_main.json`. It normalizes the JSON to standard keys (`entity_id`, `name`, `url`, etc.) and maps the raw `"keys"` field to a standard SHL `"test_type"` (e.g., Knowledge & Skills -> "K"). It also concatenates all descriptive text into a `search_text` block to support vector search. This ensures the data is only read once into memory, minimizing latency per request.

## 4. Retrieval and Ranking Strategy
We implemented a **Hybrid Deterministic Retriever**:
- **TF-IDF Vectorization**: Uses `scikit-learn` to calculate semantic similarity against `search_text`.
- **Keyword & Rule Boosting**: Exact matches for roles, skills, or specific test types receive score multipliers. Constraints extracted from the conversation (e.g., job levels, duration, remote capability) apply heavy boosts to matching products. 
- **Penalization**: Pre-packaged job solutions or products with "solution" in their name are penalized in favor of individual assessment tests, matching assignment guidelines.

By relying on `scikit-learn` and rule-based boosts instead of a Vector DB or LLM-based retrieval, the system guarantees determinism and extreme speed, executing comfortably within the 30-second timeout window.

## 5. Conversation Handling & LangGraph Orchestration
The stateless architecture means the LangGraph graph is constructed and executed from scratch for each `POST /chat` invocation. The state is represented via a `TypedDict`. 
If the user intent is vague (e.g., "I need a test"), the graph routes to a `clarification_node` without performing retrieval. If the intent is `recommend` or `refine`, it performs full context extraction and retrieval. If `compare`, it fetches specific products by name. This isolated routing keeps the LLM prompt context clean and focused.

## 6. Guardrails and Refusal Logic
The `guardrails_node` inspects user messages against sets of forbidden keywords and regex patterns (legal terms, salary, discrimination, generic coding questions). If matched, the graph routes directly to a `refusal_node` which returns a hardcoded schema-compliant JSON, completely bypassing the LLM and retriever.

## 7. Use of AI Tools
**Google Gemini 2.5 Flash** is used strictly for Natural Language Generation (NLG). The retrieved catalog items are formatted into a safe "Grounded Context" string and passed to Gemini, instructing it to *only* write the conversational `reply` field. 
To prevent hallucinated URLs:
- Gemini is explicitly instructed not to generate JSON.
- The Python backend (in `response_validation_node`) constructs the final `recommendations` list exclusively from the retrieved `item["url"]` mapping, merging it with Gemini's text reply.
If the API key is missing or the request times out, deterministic fallback strings are automatically inserted.

## 8. Evaluation Approach
Evaluation is embedded into the test suite via `pytest`. We validate:
1. Strict schema compliance for all response branches.
2. Groundedness (all returned URLs must exist in `data/shl_catalog_main.json`).
3. Correct routing for vague queries (empty recommendations + clarify intent).
4. Safety against prompt injection and out-of-scope topics.

## 9. What Did Not Work
Initially, I considered allowing the LLM to output the entire JSON response (including recommendations). However, this introduced a risk of hallucinated URLs or missing schema fields. Offloading the JSON construction and URL mapping to Python (and restricting the LLM to just the `reply` string) proved vastly more robust and 100% compliant with the assignment constraints.
