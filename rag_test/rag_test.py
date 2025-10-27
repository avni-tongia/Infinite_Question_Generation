import json
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# 1️⃣ Load the small embedding model (fast & reliable)
# -------------------------------
print("🚀 Loading embedding model...")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# -------------------------------
# 2️⃣ Load the physics context file
# -------------------------------
print("📚 Loading physics context...")
with open("physics.context.txt", "r", encoding="utf-8") as f:

    context = f.read().split("\n")

# Remove empty lines if any
context = [line.strip() for line in context if line.strip()]

# -------------------------------
# 3️⃣ Encode the context for retrieval
# -------------------------------
print("🧠 Encoding context paragraphs...")
context_embeddings = model.encode(context, convert_to_tensor=True)

# -------------------------------
# 4️⃣ Load sample questions
# -------------------------------
print("❓ Loading sample questions...")
with open("sample_queries.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

# -------------------------------
# 5️⃣ Perform retrieval for each question
# -------------------------------
print("🔍 Performing retrieval...")
results = []

for q in questions:
    query = q["question"]
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, context_embeddings, top_k=3)[0]
    top_contexts = [context[hit["corpus_id"]] for hit in hits]
    results.append({
        "question": query,
        "retrieved_contexts": top_contexts
    })

# -------------------------------
# 6️⃣ Save RAG results to JSON
# -------------------------------
print("💾 Saving results to rag_results.json ...")
with open("rag_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n✅ Retrieval complete! Check rag_results.json for outputs.")
