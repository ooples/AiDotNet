# GraphRAG - Knowledge Graph-Enhanced RAG

This sample demonstrates Knowledge Graph-Enhanced Retrieval-Augmented Generation (GraphRAG) using AiDotNet, combining vector similarity search with structured graph traversal for richer context retrieval.

## What You'll Learn

- How to build and populate a knowledge graph
- How to extract entities and relations from text
- How to perform graph-based retrieval (traversal, path finding)
- How to combine vector similarity with graph context (hybrid retrieval)
- How GraphRAG provides richer context than traditional RAG

## What is GraphRAG?

Traditional RAG finds similar documents using vector search. GraphRAG goes further:

```
Traditional RAG:
  Query -> Vector Search -> Similar Documents -> LLM -> Answer

GraphRAG:
  Query -> Entity Extraction -> Graph Traversal -> Related Entities
                             -> Vector Search   -> Similar Documents
                                                -> Combined Context -> LLM -> Answer
```

GraphRAG provides richer context by exploring relationships between concepts.

## Running the Sample

```bash
cd samples/nlp/RAG/GraphRAG
dotnet run
```

## Expected Output

```
=== AiDotNet Graph RAG ===
Knowledge Graph-Enhanced Retrieval-Augmented Generation

Building Knowledge Graph...

  Added 17 concept nodes
  Added 22 relationships

Knowledge Graph Statistics:
  - Nodes: 17
  - Edges: 22

Graph Structure:
------------------------------------------------------------

ARCHITECTURE:
  - Transformers
      --[USED_IN]--> Natural Language Processing
      --[TYPE_OF]--> Neural Networks
  - Convolutional Neural Networks
      --[USED_IN]--> Computer Vision
      --[TYPE_OF]--> Neural Networks
  ...

CONCEPT:
  - Machine Learning
  - Deep Learning
      --[SUBSET_OF]--> Machine Learning
  - Natural Language Processing
      --[USES]--> Deep Learning
      --[SUBSET_OF]--> Machine Learning
  ...

============================================================
Entity and Relation Extraction
============================================================

Extracting entities and relations from text:

Text: "Geoffrey Hinton, often called the godfather of AI, developed..."

Extracted:
  Entities:
    - PERSON: Geoffrey Hinton
    - ORGANIZATION: University of Toronto
    - TECHNOLOGY: backpropagation
  Relations:
    - Geoffrey Hinton --[DEVELOPED]--> backpropagation
    - Geoffrey Hinton --[AFFILIATED_WITH]--> University of Toronto

============================================================
Graph-Based Retrieval
============================================================

Query: "What techniques are used in deep learning?"
--------------------------------------------------

Identified entities in query:
  - CONCEPT: Deep Learning

Graph traversal (1-hop neighborhood):
  Retrieved 8 related entities:
    - [CONCEPT] Machine Learning
    - [CONCEPT] Natural Language Processing
    - [CONCEPT] Computer Vision
    - [CONCEPT] Neural Networks
    - [TECHNIQUE] Backpropagation
    ...

Generated context for LLM:
  Deep Learning: Machine learning using neural networks with multiple layers.
  Deep Learning subset of Machine Learning. Neural Networks used in Deep Learning.

============================================================
Path Finding in Knowledge Graph
============================================================

Question: How is deep learning connected to NLP?
Path found (3 hops):
  Deep Learning --[SUBSET_OF]-->
  Machine Learning --[?]-->
  Natural Language Processing

============================================================
Hybrid Retrieval: Vector Similarity + Graph Context
============================================================

Query: "What are modern techniques for understanding language?"

1. Vector Similarity Results:
   [0.7234] Natural Language Processing
   [0.6891] Transformers
   [0.6543] Attention Mechanism

2. Graph-Expanded Context:
   Expanded from 3 to 12 relevant entities

3. Final Ranked Results (Hybrid):
   [0.9234] Natural Language Processing
            AI for understanding human language...
   [0.8891] Transformers
            Neural architecture using self-attention...
   [0.7654] BERT
            Bidirectional Encoder Representations...
```

## How It Works

### 1. Knowledge Graph Structure

```
Nodes (Entities):
  - CONCEPT: Machine Learning, Deep Learning, NLP...
  - ARCHITECTURE: Transformers, CNN, RNN...
  - TECHNIQUE: Attention, Backpropagation...
  - MODEL: GPT, BERT...

Edges (Relationships):
  - Deep Learning --[SUBSET_OF]--> Machine Learning
  - Transformers --[USED_IN]--> NLP
  - GPT --[BASED_ON]--> Transformers
```

### 2. Entity and Relation Extraction

```csharp
// Extract entities and relations from text
var (entities, relations) = ExtractEntitiesAndRelations(text);

// Example result:
// Entities: [(Geoffrey Hinton, PERSON), (GPT-4, TECHNOLOGY)]
// Relations: [(Geoffrey Hinton, DEVELOPED, GPT-4)]
```

### 3. Graph Traversal

```csharp
// Find neighbors (1-hop)
var neighbors = knowledgeGraph.GetNeighbors(entity.Id);

// Find shortest path
var path = knowledgeGraph.FindShortestPath("deep_learning", "nlp");

// Breadth-first traversal
var nodes = knowledgeGraph.BreadthFirstTraversal(startNode, maxDepth: 2);
```

### 4. Hybrid Retrieval

Combines vector similarity with graph structure:

```csharp
// 1. Vector search for initial candidates
var vectorResults = documents.OrderByDescending(d => CosineSimilarity(query, d));

// 2. Expand with graph neighbors
foreach (var result in topResults)
{
    expandedContext.UnionWith(graph.GetNeighbors(result.Id));
}

// 3. Re-rank with combined score
finalScore = vectorScore * 0.7 + graphScore * 0.3;
```

## Key Components

### GraphNode

```csharp
var node = new GraphNode<float>("deep_learning", "CONCEPT");
node.SetProperty("name", "Deep Learning");
node.SetProperty("description", "ML using neural networks with multiple layers");
node.Embedding = embeddingModel.Embed("Deep Learning description...");
```

### GraphEdge

```csharp
var edge = new GraphEdge<float>(
    sourceId: "deep_learning",
    targetId: "machine_learning",
    relationType: "SUBSET_OF",
    weight: 0.95
);
```

### KnowledgeGraph

```csharp
var graph = new KnowledgeGraph<float>(new MemoryGraphStore<float>());
graph.AddNode(node);
graph.AddEdge(edge);
var neighbors = graph.GetNeighbors(nodeId);
var path = graph.FindShortestPath(startId, endId);
```

## Why GraphRAG?

| Feature | Traditional RAG | GraphRAG |
|---------|----------------|----------|
| Retrieval | Vector similarity | Vector + Graph traversal |
| Context | Similar documents | Related entities and relationships |
| Reasoning | Implicit in LLM | Explicit in graph structure |
| Multi-hop | Limited | Natural via path finding |
| Explainability | Low | High (visible relationships) |

## Use Cases

1. **Question Answering**: "How is X related to Y?" - use path finding
2. **Fact Verification**: Check if relationships exist in graph
3. **Recommendation**: Find similar entities through shared connections
4. **Exploration**: "What else is related to this topic?"

## Architecture

```
                   Query
                     |
          +----------+-----------+
          |                      |
          v                      v
   +-------------+      +----------------+
   | Entity      |      | Vector Search  |
   | Extraction  |      | (Embeddings)   |
   +------+------+      +-------+--------+
          |                     |
          v                     v
   +-------------+      +----------------+
   | Graph       |      | Top-K Results  |
   | Traversal   |      +-------+--------+
   +------+------+              |
          |                     |
          +----------+----------+
                     |
                     v
            +-----------------+
            | Hybrid Ranking  |
            | (Combine Scores)|
            +--------+--------+
                     |
                     v
            +-----------------+
            | Context for LLM |
            +-----------------+
```

## Production Considerations

1. **Graph Storage**: Use `FileGraphStore` or database backend for persistence
2. **Entity Extraction**: Use NER models (spaCy, HuggingFace) for production
3. **Relation Extraction**: Use relation extraction models or LLMs
4. **Scaling**: Consider graph databases (Neo4j) for large graphs
5. **Embeddings**: Use production embedding models (not StubEmbeddingModel)

## Next Steps

- [BasicRAG](../BasicRAG/) - Simpler vector-only RAG
- [Embeddings](../../Embeddings/) - Learn about text embeddings
- [TextClassification](../../TextClassification/) - Classify text into categories
