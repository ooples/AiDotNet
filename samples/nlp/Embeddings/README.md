# Text Embeddings - Semantic Similarity Search

This sample demonstrates how to use text embeddings for semantic similarity search and vector operations using AiDotNet.

## What You'll Learn

- How to generate vector embeddings from text
- How to compute cosine similarity between embeddings
- How to perform semantic similarity search
- How embedding arithmetic works (e.g., "king - man + woman = queen")
- How to efficiently batch-process multiple texts

## What are Embeddings?

Embeddings convert text into dense numerical vectors that capture semantic meaning:

```
"The cat sat on the mat"  -->  [0.23, -0.45, 0.78, ..., 0.12]
                                 ^
                           384-dimensional vector
```

Similar meanings produce similar vectors, enabling semantic search.

## Running the Sample

```bash
cd samples/nlp/Embeddings
dotnet run
```

## Expected Output

```
=== AiDotNet Text Embeddings ===
Semantic Similarity Search with Vector Embeddings

Embedding Model Configuration:
  - Dimension: 384
  - Max Tokens: 512

Embedding 10 documents...

  [1] "Machine learning is a subset of artificial intellig..."
  [2] "Deep learning uses neural networks with many layers..."
  ...

  Generated 10 embeddings of dimension 384

============================================================
Semantic Similarity Search
============================================================

Query: "How do neural networks learn from examples?"
--------------------------------------------------

Top 3 Similar Documents:
  1. [0.8234] "Deep learning uses neural networks with many layers..."
  2. [0.7891] "Machine learning is a subset of artificial intellig..."
  3. [0.6543] "Reinforcement learning trains agents to make decisi..."

Query: "What are some applications of AI in understanding text?"
--------------------------------------------------

Top 3 Similar Documents:
  1. [0.8012] "Natural language processing enables computers to und..."
  2. [0.6234] "Machine learning is a subset of artificial intellig..."
  3. [0.5891] "Deep learning uses neural networks with many layers..."

============================================================
Pairwise Similarity Analysis
============================================================

AI/ML Documents (indices 0-4):
  [0] Machine learning is a subset of artificial intellig...
  [1] Deep learning uses neural networks with many layers...
  ...

Other Topics (indices 5-9):
  [5] The weather today is sunny with a high of 75 degre...
  [6] Pizza is one of the most popular foods in the worl...
  ...

Similarity Analysis:
  - Within AI/ML group:    0.7234
  - Within Other group:    0.3456
  - Between groups:        0.2891

Interpretation:
  AI/ML documents are more similar to each other than to other topics.
  This demonstrates that embeddings capture semantic meaning!

============================================================
Embedding Arithmetic (Vector Operations)
============================================================

Demonstrating: "Deep Learning" - "Learning" + "Vision" ~ "Computer Vision"

Similarity of result to 'Computer Vision': 0.7891
Most similar document: [3] "Computer vision allows machines to interpret and..."
```

## How It Works

### 1. Generating Embeddings

```csharp
// Create embedding model
var embeddingModel = new StubEmbeddingModel<float>(embeddingDimension: 384);

// Embed a single text
Vector<float> embedding = embeddingModel.Embed("Machine learning is amazing");

// Batch embed multiple texts (more efficient)
Matrix<float> embeddings = embeddingModel.EmbedBatch(new[] { "text1", "text2", "text3" });
```

### 2. Computing Cosine Similarity

```csharp
double CosineSimilarity(Vector<float> a, Vector<float> b)
{
    // cosine_sim = (a . b) / (||a|| * ||b||)
    double dotProduct = Sum(a[i] * b[i]);
    double normA = Sqrt(Sum(a[i]^2));
    double normB = Sqrt(Sum(b[i]^2));
    return dotProduct / (normA * normB);
}
```

**Similarity Scale:**
- 1.0: Identical meaning
- 0.7-0.9: Very similar
- 0.4-0.7: Somewhat related
- 0.0-0.4: Different topics
- -1.0: Opposite meaning

### 3. Semantic Search

```csharp
// Embed query
var queryEmbedding = embeddingModel.Embed("How do neural networks work?");

// Find most similar documents
var results = documents
    .Select((doc, i) => new { Index = i, Similarity = CosineSimilarity(queryEmbedding, docEmbeddings[i]) })
    .OrderByDescending(r => r.Similarity)
    .Take(3);
```

### 4. Embedding Arithmetic

Famous example: `king - man + woman = queen`

```csharp
// Vector arithmetic
var result = kingEmb - manEmb + womanEmb;

// Find closest word
var mostSimilar = FindMostSimilar(result, vocabulary);  // Returns "queen"
```

## Available Embedding Models

| Model | Description | Dimension |
|-------|-------------|-----------|
| `StubEmbeddingModel` | Hash-based (for testing) | Configurable |
| `HuggingFaceEmbeddingModel` | HuggingFace models | Model-dependent |
| `OpenAIEmbeddingModel` | OpenAI ada-002 | 1536 |
| `CohereEmbeddingModel` | Cohere embeddings | 1024/4096 |
| `LocalTransformerEmbedding` | Local transformer | Model-dependent |

## Production Usage

For production, replace the stub model with a real embedding model:

```csharp
// Option 1: HuggingFace (local or API)
var model = new HuggingFaceEmbeddingModel<float>("sentence-transformers/all-MiniLM-L6-v2");

// Option 2: OpenAI
var model = new OpenAIEmbeddingModel<float>(apiKey: "your-key");

// Option 3: Cohere
var model = new CohereEmbeddingModel<float>(apiKey: "your-key");
```

## Use Cases

1. **Semantic Search**: Find documents by meaning, not just keywords
2. **Duplicate Detection**: Identify similar content
3. **Clustering**: Group related documents together
4. **Recommendation**: Suggest similar items
5. **RAG**: Retrieve context for language models

## Architecture

```
        Input Text
            |
            v
    +--------------+
    | Tokenization |
    +--------------+
            |
            v
    +--------------+
    | Embedding    |
    | Model        |
    +--------------+
            |
            v
    +--------------+
    | Dense Vector |
    | [dim=384]    |
    +--------------+
            |
            v
    +--------------+
    | Cosine       |
    | Similarity   |
    +--------------+
            |
            v
      Similarity Score
```

## Next Steps

- [BasicRAG](../RAG/BasicRAG/) - Use embeddings for retrieval-augmented generation
- [GraphRAG](../RAG/GraphRAG/) - Knowledge graph-enhanced retrieval
- [TextClassification](../TextClassification/) - Classify text into categories
