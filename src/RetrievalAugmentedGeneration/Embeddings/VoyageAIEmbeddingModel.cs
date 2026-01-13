using System;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;

namespace AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels;

/// <summary>
/// Voyage AI-compatible embedding model using ONNX for high-performance local inference.
/// </summary>
/// <typeparam name="T">The numeric data type used for vector operations.</typeparam>
public class VoyageAIEmbeddingModel<T> : EmbeddingModelBase<T>
{
    private const int DefaultMaxTokens = 16000;
    private readonly string _apiKey;
    private readonly string _model;
    private readonly string _inputType;
    private readonly int _dimension;
    private readonly ONNXSentenceTransformer<T> _onnxTransformer;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the <see cref="VoyageAIEmbeddingModel{T}"/> class.
    /// </summary>
    /// <param name="apiKey">The Voyage AI API key (unused for local ONNX).</param>
    /// <param name="model">The model path (e.g., "path/to/voyage-model.onnx").</param>
    /// <param name="inputType">The input type ("document" or "query").</param>
    /// <param name="dimension">The embedding dimension.</param>
    public VoyageAIEmbeddingModel(
        string apiKey,
        string model,
        string inputType,
        int dimension)
    {
#pragma warning disable CS0414 // Field is assigned but its value is never used
        _apiKey = apiKey; // Kept for API compatibility but unused
#pragma warning restore CS0414
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _inputType = inputType ?? throw new ArgumentNullException(nameof(inputType));
        _dimension = dimension;

        // Initialize ONNX transformer once
        _onnxTransformer = new ONNXSentenceTransformer<T>(
            modelPath: _model,
            dimension: _dimension,
            maxTokens: DefaultMaxTokens
        );
    }

    /// <inheritdoc />
    public override int EmbeddingDimension => _dimension;

    /// <inheritdoc />
    public override int MaxTokens => DefaultMaxTokens;

    /// <inheritdoc />
    protected override Vector<T> EmbedCore(string text)
    {
        return _onnxTransformer.Embed(text);
    }

    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _onnxTransformer.Dispose();
            }
            _disposed = true;
        }
        base.Dispose(disposing);
    }
}


