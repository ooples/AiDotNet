using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using Vector = AiDotNet.Tensors.LinearAlgebra.Vector<float>;

Console.WriteLine("=== AiDotNet Text Embeddings ===");
Console.WriteLine("Semantic Similarity Search with Vector Embeddings\n");

// Create embedding model (using stub for demo - replace with real model in production)
var embeddingModel = new StubEmbeddingModel<float>(embeddingDimension: 384, maxTokens: 512);

Console.WriteLine($"Embedding Model Configuration:");
Console.WriteLine($"  - Dimension: {embeddingModel.EmbeddingDimension}");
Console.WriteLine($"  - Max Tokens: {embeddingModel.MaxTokens}");
Console.WriteLine();

// Sample documents to embed
var documents = new[]
{
    "Machine learning is a subset of artificial intelligence focused on building systems that learn from data.",
    "Deep learning uses neural networks with many layers to model complex patterns in data.",
    "Natural language processing enables computers to understand and generate human language.",
    "Computer vision allows machines to interpret and understand visual information from images and videos.",
    "Reinforcement learning trains agents to make decisions by rewarding desired behaviors.",
    "The weather today is sunny with a high of 75 degrees Fahrenheit.",
    "Pizza is one of the most popular foods in the world, originating from Italy.",
    "The stock market closed higher today with technology stocks leading the gains.",
    "Regular exercise and a balanced diet are essential for maintaining good health.",
    "The ancient pyramids of Egypt were built over 4,500 years ago."
};

Console.WriteLine($"Embedding {documents.Length} documents...\n");

// Generate embeddings for all documents
var documentEmbeddings = new Vector[documents.Length];
for (int i = 0; i < documents.Length; i++)
{
    documentEmbeddings[i] = embeddingModel.Embed(documents[i]);
    Console.WriteLine($"  [{i + 1}] \"{documents[i][..Math.Min(50, documents[i].Length)]}...\"");
}

Console.WriteLine($"\n  Generated {documents.Length} embeddings of dimension {embeddingModel.EmbeddingDimension}");

// Semantic similarity search demonstration
Console.WriteLine("\n" + new string('=', 60));
Console.WriteLine("Semantic Similarity Search");
Console.WriteLine(new string('=', 60));

var queries = new[]
{
    "How do neural networks learn from examples?",
    "What are some applications of AI in understanding text?",
    "Tell me about healthy lifestyle choices",
    "What is the history of ancient monuments?"
};

foreach (var query in queries)
{
    Console.WriteLine($"\nQuery: \"{query}\"");
    Console.WriteLine(new string('-', 50));

    // Embed the query
    var queryEmbedding = embeddingModel.Embed(query);

    // Compute cosine similarity with all documents
    var similarities = new List<(int index, double similarity)>();
    for (int i = 0; i < documents.Length; i++)
    {
        double similarity = CosineSimilarity(queryEmbedding, documentEmbeddings[i]);
        similarities.Add((i, similarity));
    }

    // Sort by similarity (descending) and show top 3
    var topResults = similarities.OrderByDescending(s => s.similarity).Take(3).ToList();

    Console.WriteLine("\nTop 3 Similar Documents:");
    for (int rank = 0; rank < topResults.Count; rank++)
    {
        var (index, similarity) = topResults[rank];
        Console.WriteLine($"  {rank + 1}. [{similarity:F4}] \"{documents[index][..Math.Min(60, documents[index].Length)]}...\"");
    }
}

// Pairwise similarity matrix demonstration
Console.WriteLine("\n" + new string('=', 60));
Console.WriteLine("Pairwise Similarity Analysis");
Console.WriteLine(new string('=', 60));

// Group documents by topic for analysis
var aiDocs = new[] { 0, 1, 2, 3, 4 };  // AI/ML related
var otherDocs = new[] { 5, 6, 7, 8, 9 };  // Other topics

Console.WriteLine("\nAI/ML Documents (indices 0-4):");
foreach (var i in aiDocs)
{
    Console.WriteLine($"  [{i}] {documents[i][..Math.Min(50, documents[i].Length)]}...");
}

Console.WriteLine("\nOther Topics (indices 5-9):");
foreach (var i in otherDocs)
{
    Console.WriteLine($"  [{i}] {documents[i][..Math.Min(50, documents[i].Length)]}...");
}

// Compute average within-group similarity
double aiGroupSimilarity = ComputeAverageGroupSimilarity(documentEmbeddings, aiDocs);
double otherGroupSimilarity = ComputeAverageGroupSimilarity(documentEmbeddings, otherDocs);
double crossGroupSimilarity = ComputeAverageCrossGroupSimilarity(documentEmbeddings, aiDocs, otherDocs);

Console.WriteLine("\nSimilarity Analysis:");
Console.WriteLine($"  - Within AI/ML group:    {aiGroupSimilarity:F4}");
Console.WriteLine($"  - Within Other group:    {otherGroupSimilarity:F4}");
Console.WriteLine($"  - Between groups:        {crossGroupSimilarity:F4}");

Console.WriteLine("\nInterpretation:");
if (aiGroupSimilarity > crossGroupSimilarity)
{
    Console.WriteLine("  AI/ML documents are more similar to each other than to other topics.");
    Console.WriteLine("  This demonstrates that embeddings capture semantic meaning!");
}

// Embedding arithmetic demonstration
Console.WriteLine("\n" + new string('=', 60));
Console.WriteLine("Embedding Arithmetic (Vector Operations)");
Console.WriteLine(new string('=', 60));

Console.WriteLine("\nDemonstrating: \"Deep Learning\" - \"Learning\" + \"Vision\" ~ \"Computer Vision\"");

var deepLearningEmb = embeddingModel.Embed("Deep learning models neural networks");
var learningEmb = embeddingModel.Embed("Learning from data and examples");
var visionEmb = embeddingModel.Embed("Visual information and images");
var computerVisionEmb = embeddingModel.Embed("Computer vision processes images");

// Compute: deep_learning - learning + vision
var resultEmb = VectorArithmetic(deepLearningEmb, learningEmb, visionEmb);

// Normalize result
resultEmb = NormalizeVector(resultEmb);

// Compare with computer vision embedding
double analogySimilarity = CosineSimilarity(resultEmb, computerVisionEmb);
Console.WriteLine($"\nSimilarity of result to 'Computer Vision': {analogySimilarity:F4}");

// Find most similar document to the computed vector
var analogyResults = new List<(int index, double similarity)>();
for (int i = 0; i < documents.Length; i++)
{
    double sim = CosineSimilarity(resultEmb, documentEmbeddings[i]);
    analogyResults.Add((i, sim));
}

var topAnalogy = analogyResults.OrderByDescending(s => s.similarity).First();
Console.WriteLine($"Most similar document: [{topAnalogy.index}] \"{documents[topAnalogy.index][..Math.Min(50, documents[topAnalogy.index].Length)]}...\"");

// Batch embedding demonstration
Console.WriteLine("\n" + new string('=', 60));
Console.WriteLine("Batch Embedding (Efficient Processing)");
Console.WriteLine(new string('=', 60));

var batchTexts = new[]
{
    "Transformers revolutionized NLP",
    "Attention mechanisms focus on relevant parts",
    "BERT uses bidirectional context"
};

Console.WriteLine($"\nBatch embedding {batchTexts.Length} texts...");

var batchMatrix = embeddingModel.EmbedBatch(batchTexts);
Console.WriteLine($"Result: Matrix of shape [{batchTexts.Length}, {embeddingModel.EmbeddingDimension}]");

Console.WriteLine("\nBatch texts and their embedding norms:");
for (int i = 0; i < batchTexts.Length; i++)
{
    double norm = 0;
    for (int j = 0; j < embeddingModel.EmbeddingDimension; j++)
    {
        double val = Convert.ToDouble(batchMatrix[i, j]);
        norm += val * val;
    }
    norm = Math.Sqrt(norm);
    Console.WriteLine($"  [{i}] \"{batchTexts[i]}\" - L2 Norm: {norm:F4}");
}

Console.WriteLine("\n=== Sample Complete ===");

// Helper functions
static double CosineSimilarity(Vector a, Vector b)
{
    if (a.Length != b.Length)
        throw new ArgumentException("Vectors must have the same length");

    double dotProduct = 0;
    double normA = 0;
    double normB = 0;

    for (int i = 0; i < a.Length; i++)
    {
        double valA = Convert.ToDouble(a[i]);
        double valB = Convert.ToDouble(b[i]);
        dotProduct += valA * valB;
        normA += valA * valA;
        normB += valB * valB;
    }

    double denominator = Math.Sqrt(normA) * Math.Sqrt(normB);
    return denominator > 0 ? dotProduct / denominator : 0;
}

static double ComputeAverageGroupSimilarity(Vector[] embeddings, int[] indices)
{
    if (indices.Length < 2) return 1.0;

    double totalSimilarity = 0;
    int count = 0;

    for (int i = 0; i < indices.Length; i++)
    {
        for (int j = i + 1; j < indices.Length; j++)
        {
            totalSimilarity += CosineSimilarity(embeddings[indices[i]], embeddings[indices[j]]);
            count++;
        }
    }

    return count > 0 ? totalSimilarity / count : 0;
}

static double ComputeAverageCrossGroupSimilarity(Vector[] embeddings, int[] group1, int[] group2)
{
    double totalSimilarity = 0;
    int count = 0;

    foreach (var i in group1)
    {
        foreach (var j in group2)
        {
            totalSimilarity += CosineSimilarity(embeddings[i], embeddings[j]);
            count++;
        }
    }

    return count > 0 ? totalSimilarity / count : 0;
}

static Vector VectorArithmetic(Vector a, Vector subtract, Vector add)
{
    var result = new float[a.Length];
    for (int i = 0; i < a.Length; i++)
    {
        result[i] = a[i] - subtract[i] + add[i];
    }
    return new Vector(result);
}

static Vector NormalizeVector(Vector vector)
{
    double magnitude = 0;
    for (int i = 0; i < vector.Length; i++)
    {
        double val = Convert.ToDouble(vector[i]);
        magnitude += val * val;
    }
    magnitude = Math.Sqrt(magnitude);

    if (magnitude < 1e-8) return vector;

    var normalized = new float[vector.Length];
    for (int i = 0; i < vector.Length; i++)
    {
        normalized[i] = (float)(Convert.ToDouble(vector[i]) / magnitude);
    }
    return new Vector(normalized);
}
