using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;

namespace AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels;

/// <summary>
/// Voyage AI-compatible embedding model using ONNX for high-performance local inference.
/// </summary>
/// <typeparam name="T">The numeric data type used for vector operations.</typeparam>
/// <remarks>
/// <para>
/// This implementation provides Voyage AI-compatible embeddings using local ONNX inference instead of
/// external API calls. Voyage AI models are specifically optimized for retrieval tasks and achieve
/// state-of-the-art performance on benchmarks like MTEB (Massive Text Embedding Benchmark).
/// </para>
/// <para><b>For Beginners:</b> What are embeddings and why Voyage AI?
/// 
/// Embeddings are numerical representations of text that capture meaning:
/// - "cat" → [0.2, 0.8, 0.1, ...] (768 numbers)
/// - "kitten" → [0.21, 0.79, 0.09, ...] (very similar numbers = similar meaning)
/// - "car" → [-0.5, 0.1, 0.9, ...] (different numbers = different meaning)
/// 
/// Why Voyage AI?
/// - Specialized for search/retrieval (not general-purpose)
/// - Better at distinguishing relevant from irrelevant documents
/// - Supports longer documents (up to 16,000 tokens)
/// - Separate optimizations for "query" vs "document" embeddings
/// - Industry-leading benchmark scores
/// 
/// Current Implementation:
/// Instead of calling Voyage AI's API, this uses ONNX (Open Neural Network Exchange) to run
/// Voyage-compatible models locally on your machine. Benefits:
/// - No API costs
/// - No network latency
/// - Works offline
/// - Data privacy (everything stays local)
/// </para>
/// <para><b>Example Usage:</b>
/// <code>
/// // Option 1: Local ONNX inference (current implementation)
/// var model = new VoyageAIEmbeddingModel&lt;double&gt;(
///     apiKey: "not-used",  // Not needed for ONNX
///     model: "path/to/voyage-model.onnx",  // Download from HuggingFace
///     inputType: "document",  // or "query"
///     dimension: 1024
/// );
/// 
/// // Embed documents
/// var docEmbedding = model.Embed("Photosynthesis converts light into chemical energy");
/// 
/// // Embed queries (would use different model with inputType="query")
/// var queryEmbedding = model.Embed("How does photosynthesis work?");
/// 
/// // Calculate similarity
/// var similarity = Vector.CosineSimilarity(queryEmbedding, docEmbedding);
/// // High similarity = relevant document!
/// </code>
/// </para>
/// <para><b>How It Works:</b>
/// Internal process:
/// 1. Text input → Tokenization (break into words/subwords)
/// 2. Tokens → ONNX model (neural network processing)
/// 3. Model output → Dense vector of numbers (embedding)
/// 4. Return embedding for similarity calculations
/// 
/// InputType matters:
/// - "document": Optimized for indexing content (broader matching)
/// - "query": Optimized for search queries (precise matching)
/// - Use "document" type when embedding your document store
/// - Use "query" type when embedding user search queries
/// </para>
/// <para><b>Benefits:</b>
/// - State-of-the-art retrieval performance
/// - Long context support (16K tokens)
/// - Optimized for asymmetric search (queries vs documents)
/// - Local inference (no API dependency)
/// - Privacy-preserving (data never leaves your system)
/// - Cost-effective (no per-request charges)
/// </para>
/// <para><b>Limitations:</b>
/// - Requires ONNX model file (must download from HuggingFace)
/// - Model file can be large (300MB-2GB depending on variant)
/// - First inference slower due to model loading
/// - CPU inference slower than GPU
/// - No automatic model updates (must manually download new versions)
/// </para>
/// </remarks>
public class VoyageAIEmbeddingModel<T> : EmbeddingModelBase<T>
{
    private readonly string _apiKey;
    private readonly string _model;
    private readonly string _inputType;

    private readonly int _dimension;

    /// <summary>
    /// Initializes a new instance of the <see cref="VoyageAIEmbeddingModel{T}"/> class.
    /// </summary>
    /// <param name="apiKey">The Voyage AI API key.</param>
    /// <param name="model">The model name (e.g., "voyage-02").</param>
    /// <param name="inputType">The input type ("document" or "query").</param>
    /// <param name="dimension">The embedding dimension.</param>
    public VoyageAIEmbeddingModel(
        string apiKey,
        string model,
        string inputType,
        int dimension)
    {
        _apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _inputType = inputType ?? throw new ArgumentNullException(nameof(inputType));
        _dimension = dimension;
    }

    /// <inheritdoc />
    public override int EmbeddingDimension => _dimension;

    /// <inheritdoc />
    public override int MaxTokens => 16000;

    /// <inheritdoc />
    protected override Vector<T> EmbedCore(string text)
    {
        // Use ONNXSentenceTransformer as backend instead of external API
        // This provides local inference with ONNX models
        var onnxTransformer = new ONNXSentenceTransformer<T>(
            modelPath: _model,  // Model path instead of API model name
            dimension: _dimension,
            maxTokens: MaxTokens
        );

        return onnxTransformer.Embed(text);
    }


}


