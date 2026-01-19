using AiDotNet;
using AiDotNet.RetrievalAugmentedGeneration;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using AiDotNet.RetrievalAugmentedGeneration.Rerankers;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using AiDotNet.RetrievalAugmentedGeneration.VectorStores;

Console.WriteLine("=== AiDotNet Basic RAG ===");
Console.WriteLine("Building a Question-Answering System with Retrieval-Augmented Generation\n");

// Sample documents about AI/ML
var documents = new[]
{
    new Document
    {
        Id = "doc1",
        Content = "Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.",
        Metadata = new Dictionary<string, object> { ["topic"] = "ML Basics" }
    },
    new Document
    {
        Id = "doc2",
        Content = "Deep learning is a type of machine learning based on artificial neural networks with multiple layers (hence 'deep'). These neural networks attempt to simulate the behavior of the human brain in processing data for tasks such as speech recognition, image recognition, and natural language processing.",
        Metadata = new Dictionary<string, object> { ["topic"] = "Deep Learning" }
    },
    new Document
    {
        Id = "doc3",
        Content = "Supervised learning is a type of machine learning where the model is trained on labeled data. The algorithm learns to map input features to output labels by finding patterns in the training data. Common examples include classification and regression tasks.",
        Metadata = new Dictionary<string, object> { ["topic"] = "Supervised Learning" }
    },
    new Document
    {
        Id = "doc4",
        Content = "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language. NLP combines computational linguistics with machine learning to process and analyze large amounts of natural language data.",
        Metadata = new Dictionary<string, object> { ["topic"] = "NLP" }
    },
    new Document
    {
        Id = "doc5",
        Content = "Reinforcement learning is an area of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward. Unlike supervised learning, RL does not require labeled input/output pairs and focuses on finding a balance between exploration and exploitation.",
        Metadata = new Dictionary<string, object> { ["topic"] = "Reinforcement Learning" }
    }
};

Console.WriteLine($"Loaded {documents.Length} documents about AI/ML topics\n");

// Display document topics
Console.WriteLine("Document Topics:");
foreach (var doc in documents)
{
    Console.WriteLine($"  - {doc.Metadata["topic"]}: {doc.Content[..Math.Min(50, doc.Content.Length)]}...");
}
Console.WriteLine();

try
{
    // Build the RAG pipeline
    Console.WriteLine("Building RAG pipeline...");
    Console.WriteLine("  - Vector store: In-memory");
    Console.WriteLine("  - Retriever: Dense retriever (top-k=3)");
    Console.WriteLine("  - Reranker: Cross-encoder");
    Console.WriteLine("  - Generator: LLM-based\n");

    // Configure RAG components
    var vectorStore = new InMemoryVectorStore<float>();
    var retriever = new DenseRetriever<float>(vectorStore, topK: 3);
    var reranker = new CrossEncoderReranker<float>();

    // Build the RAG model
    var builder = new AiModelBuilder<float, string, string>()
        .ConfigureRetrievalAugmentedGeneration(
            retriever: retriever,
            reranker: reranker,
            generator: null);  // Generator would be configured with actual LLM

    Console.WriteLine("Indexing documents...");

    // In a real implementation, this would:
    // 1. Embed each document using an embedding model
    // 2. Store embeddings in the vector store

    // Simulate indexing
    foreach (var doc in documents)
    {
        // vectorStore.Add(doc.Id, embedding, doc);
    }

    Console.WriteLine($"  Indexed {documents.Length} documents\n");

    // Demo queries
    var queries = new[]
    {
        "What is machine learning?",
        "How does deep learning work?",
        "What is the difference between supervised and reinforcement learning?"
    };

    foreach (var query in queries)
    {
        Console.WriteLine($"Query: \"{query}\"");
        Console.WriteLine("─────────────────────────────────────");

        // Simulate retrieval (in real implementation, this uses embeddings)
        var relevantDocs = SimulateRetrieval(query, documents);

        Console.WriteLine("\nRetrieved documents:");
        foreach (var (doc, score, rank) in relevantDocs)
        {
            Console.WriteLine($"  {rank}. [{score:F2}] {doc.Content[..Math.Min(60, doc.Content.Length)]}...");
        }

        // Simulate answer generation
        var answer = SimulateGeneration(query, relevantDocs.Select(d => d.doc).ToArray());

        Console.WriteLine($"\nGenerated answer:");
        Console.WriteLine($"  {answer}");
        Console.WriteLine();
    }
}
catch (Exception ex)
{
    Console.WriteLine($"Note: Full RAG pipeline requires embedding model and LLM configuration.");
    Console.WriteLine($"This sample demonstrates the API pattern for RAG.");
    Console.WriteLine($"\nError details: {ex.Message}");
}

Console.WriteLine("=== Sample Complete ===");

// Simulate retrieval based on keyword matching (real impl uses vector similarity)
static List<(Document doc, double score, int rank)> SimulateRetrieval(string query, Document[] documents)
{
    var queryTerms = query.ToLower().Split(' ');
    var scores = documents.Select(doc =>
    {
        var docLower = doc.Content.ToLower();
        var score = queryTerms.Count(term => docLower.Contains(term)) / (double)queryTerms.Length;
        return (doc, score);
    })
    .OrderByDescending(x => x.score)
    .Take(3)
    .Select((x, i) => (x.doc, x.score, i + 1))
    .ToList();

    return scores;
}

// Simulate answer generation
static string SimulateGeneration(string query, Document[] context)
{
    // In real implementation, this would call an LLM with the query and context
    // For demo, we return a simple response based on the first retrieved document
    if (context.Length == 0)
        return "I don't have enough information to answer that question.";

    var firstDoc = context[0];
    var sentences = firstDoc.Content.Split('.');
    return sentences[0].Trim() + ".";
}

// Simple document class for the sample
public class Document
{
    public string Id { get; set; } = "";
    public string Content { get; set; } = "";
    public Dictionary<string, object> Metadata { get; set; } = new();
}
