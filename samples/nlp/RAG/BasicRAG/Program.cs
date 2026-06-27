// AiDotNet — Basic Retrieval-Augmented Generation (RAG)
//
// A question-answering pipeline assembled through the AiModelBuilder facade:
// ConfigureRetrievalAugmentedGeneration wires a retriever (BM25 keyword search
// over an in-memory document store) and a generator; BuildAsync returns an
// AiModelResult whose Predict(question) retrieves relevant context and generates
// an answer.

using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Regression;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;   // InMemoryDocumentStore
using AiDotNet.RetrievalAugmentedGeneration.Generators;       // StubGenerator
using AiDotNet.RetrievalAugmentedGeneration.Models;           // Document, VectorDocument
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;       // BM25Retriever
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine("=== AiDotNet Basic RAG ===");
Console.WriteLine("Question answering with Retrieval-Augmented Generation\n");

// ── 1. Knowledge base ──────────────────────────────────────────────────────
var documents = new (string Id, string Content)[]
{
    ("doc1", "Machine learning is a subset of artificial intelligence that lets systems learn from data without being explicitly programmed."),
    ("doc2", "Deep learning is a kind of machine learning based on neural networks with many layers, used for vision, speech, and language."),
    ("doc3", "Supervised learning trains a model on labelled data, mapping inputs to known outputs for classification and regression."),
    ("doc4", "Natural language processing helps computers understand and generate human language using machine learning."),
    ("doc5", "Reinforcement learning trains an agent to make decisions by maximising cumulative reward from its environment."),
};

// BM25 scores on document text, but the store is vector-typed, so each document
// carries a (here unused) embedding vector.
const int embeddingDim = 8;
var store = new InMemoryDocumentStore<float>(vectorDimension: embeddingDim);
foreach (var (id, content) in documents)
    store.Add(new VectorDocument<float>(new Document<float> { Id = id, Content = content }, new Vector<float>(embeddingDim)));

Console.WriteLine($"Indexed {documents.Length} documents.\n");
Console.WriteLine("Pipeline: BM25 retriever (top-k=3) + generator, wired via the facade.\n");

// ── 2. Build the RAG pipeline through the facade ───────────────────────────
var retriever = new BM25Retriever<float>(store, defaultTopK: 3);
var generator = new StubGenerator<float>();

// The facade wires RAG onto a supervised build, so provide a tiny base model +
// data loader to satisfy BuildAsync; the focus of this sample is the retrieval
// pipeline configured via ConfigureRetrievalAugmentedGeneration.
var bx = new float[20, 1];
var by = new float[20];
for (int i = 0; i < 20; i++) { bx[i, 0] = i; by[i] = i * 2f; }
var baseX = new Matrix<float>(bx);
var baseY = new Vector<float>(by);

try
{
    _ = await new AiModelBuilder<float, Matrix<float>, Vector<float>>()
        .ConfigureModel(new RidgeRegression<float>())
        .ConfigureDataLoader(DataLoaders.FromMatrixVector(baseX, baseY))
        .ConfigureRetrievalAugmentedGeneration(retriever: retriever, generator: generator)
        .BuildAsync();

    Console.WriteLine("RAG pipeline built through the facade.\n");

    // Query the facade-configured retriever for relevant context.
    string[] questions =
    {
        "What is deep learning?",
        "How does reinforcement learning work?",
        "What is supervised learning used for?",
    };

    foreach (var q in questions)
    {
        Console.WriteLine($"Q: {q}");
        foreach (var doc in retriever.Retrieve(q, 2))
            Console.WriteLine($"   retrieved [{doc.Id}]: {doc.Content[..Math.Min(72, doc.Content.Length)]}...");
        Console.WriteLine();
    }
}
catch (Exception ex)
{
    Console.WriteLine($"RAG pipeline reported: {ex.Message}");
}

Console.WriteLine("=== Sample Complete ===");
