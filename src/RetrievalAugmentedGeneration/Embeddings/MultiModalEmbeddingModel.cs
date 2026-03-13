using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using AiDotNet.Validation;

namespace AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels;

/// <summary>
/// Multi-modal embedding model that creates unified vector representations for both text and images in the same embedding space.
/// </summary>
/// <typeparam name="T">The numeric data type used for vector operations.</typeparam>
/// <remarks>
/// <para>
/// This model enables cross-modal similarity search by embedding text and images into the same vector space,
/// allowing you to search for images using text queries, find similar images, or retrieve text documents
/// related to images. Based on models like CLIP (Contrastive Language-Image Pre-training).
/// </para>
/// <para><b>For Beginners:</b> Imagine a search engine where you can find pictures using words, or vice versa.
/// 
/// Normal embeddings:
/// - Text embedding: "a cat" → [0.2, 0.5, 0.1, ...]
/// - Image embedding: (cat.jpg) → [0.9, 0.1, 0.3, ...] (different space, can't compare!)
/// 
/// Multi-modal embeddings (same space):
/// - Text: "a cat" → [0.2, 0.5, 0.1, ...]
/// - Image: (cat.jpg) → [0.21, 0.48, 0.12, ...] (similar values = same meaning!)
/// - Result: Can calculate similarity between text and image!
/// 
/// Real-world uses:
/// 1. Search images with text: "sunset over mountains" finds mountain sunset photos
/// 2. Search text with images: Upload product photo, find product descriptions
/// 3. Visual Q&A: "What's in this image?" matched against image embeddings
/// 4. Content recommendation: Show images similar to text user likes
/// 
/// How it works:
/// The model was trained with millions of (image, text) pairs to learn that:
/// - Image of dog + text "dog" should have similar embeddings
/// - Image of dog + text "cat" should have different embeddings
/// - This creates a "shared understanding" between vision and language
/// </para>
/// <para><b>Example Usage:</b>
/// <code>
/// // Initialize with CLIP model
/// var model = new MultiModalEmbeddingModel&lt;double&gt;(
///     modelPath: "path/to/clip-model.onnx",
///     normalizeEmbeddings: true,  // Important for cosine similarity
///     dimension: 512
/// );
/// 
/// // Embed text queries
/// var textEmbedding = model.Embed("a photo of a golden retriever");
/// 
/// // Embed images
/// var imageEmbedding = model.EmbedImage("photos/dog.jpg");
/// 
/// // Calculate cross-modal similarity
/// var similarity = Vector.CosineSimilarity(textEmbedding, imageEmbedding);
/// // High similarity means image matches text description!
/// 
/// // Batch process images
/// var imagePaths = Directory.GetFiles("photos/", "*.jpg");
/// var imageEmbeddings = model.EmbedImageBatch(imagePaths).ToList();
/// 
/// // Find best matching image for a text query
/// var query = model.Embed("brown dog playing fetch");
/// var bestMatch = imageEmbeddings
///     .Select((emb, idx) => (similarity: Vector.CosineSimilarity(query, emb), path: imagePaths[idx]))
///     .OrderByDescending(x => x.similarity)
///     .First();
/// </code>
/// </para>
/// <para><b>How It Works:</b>
/// Architecture (simplified CLIP):
/// 
/// Text Path:
/// 1. Text → Tokenizer → Token IDs
/// 2. Token IDs → Text Encoder (Transformer) → Text Features
/// 3. Text Features → Projection Head → Text Embedding (512D)
/// 
/// Image Path:
/// 1. Image → Resize/Normalize → Pixel Array
/// 2. Pixels → Image Encoder (CNN/Vision Transformer) → Image Features
/// 3. Image Features → Projection Head → Image Embedding (512D)
/// 
/// Both paths output embeddings in the same 512-dimensional space, enabling direct comparison.
/// 
/// Current implementation uses ONNX for text encoding and simulated image encoding.
/// Production should use full CLIP ONNX model with both encoders.
/// </para>
/// <para><b>Benefits:</b>
/// - Cross-modal search - Find images with text, text with images
/// - Zero-shot classification - Classify images without task-specific training
/// - Unified representation - Single embedding space for all content types
/// - Flexible retrieval - Mix text and image documents in same index
/// - Pre-trained - No custom training needed for basic use cases
/// </para>
/// <para><b>Limitations:</b>
/// - Image encoding currently simulated (hash-based) - needs real CLIP encoder
/// - Model file size can be large (1-4GB for CLIP variants)
/// - Image processing slower than text (convolutional layers are compute-intensive)
/// - Works best with natural images (struggles with abstract diagrams, charts)
/// - Limited to English text in most CLIP models (multilingual versions available)
/// </para>
/// </remarks>
public class MultiModalEmbeddingModel<T> : EmbeddingModelBase<T>
{
    private readonly string _modelPath;
    private readonly bool _normalizeEmbeddings;

    private readonly int _dimension;

    /// <summary>
    /// Initializes a new instance of the <see cref="MultiModalEmbeddingModel{T}"/> class.
    /// </summary>
    /// <param name="modelPath">Path to the multi-modal model (e.g., CLIP).</param>
    /// <param name="normalizeEmbeddings">Whether to normalize embeddings to unit length.</param>
    /// <param name="dimension">The embedding dimension.</param>
    public MultiModalEmbeddingModel(
        string modelPath,
        bool normalizeEmbeddings,
        int dimension)
    {
        Guard.NotNull(modelPath);
        _modelPath = modelPath;
        _normalizeEmbeddings = normalizeEmbeddings;
        _dimension = dimension;
    }

    /// <inheritdoc />
    public override int EmbeddingDimension => _dimension;

    /// <inheritdoc />
    public override int MaxTokens => 512;

    /// <inheritdoc />
    protected override Vector<T> EmbedCore(string text)
    {
        // Use ONNX-based sentence transformer for text embeddings
        var textModel = new ONNXSentenceTransformer<T>(_modelPath, _dimension, MaxTokens);
        var embedding = textModel.Embed(text);

        return _normalizeEmbeddings ? embedding.Normalize() : embedding;
    }

    /// <summary>
    /// Generates image embeddings from file path.
    /// </summary>
    /// <param name="imagePath">Path to the image file.</param>
    /// <returns>The embedding vector for the image.</returns>
    public Vector<T> EmbedImage(string imagePath)
    {
        if (string.IsNullOrWhiteSpace(imagePath))
            throw new ArgumentException("Image path cannot be null or whitespace", nameof(imagePath));

        if (!File.Exists(imagePath))
            throw new FileNotFoundException($"Image file not found: {imagePath}");

        // Generate embedding using image path hash
        // In production, this would use CLIP's image encoder with convolutional layers
        var values = new T[_dimension];
        var hash = GetImageHash(imagePath);

        for (int i = 0; i < _dimension; i++)
        {
            var val = NumOps.FromDouble(Math.Sin((double)hash * (i + 1) * 0.003));
            values[i] = val;
        }

        var embedding = new Vector<T>(values);
        return _normalizeEmbeddings ? embedding.Normalize() : embedding;
    }

    /// <summary>
    /// Batch embedding generation for images.
    /// </summary>
    /// <param name="imagePaths">Paths to image files.</param>
    /// <returns>Embedding vectors for all images.</returns>
    public IEnumerable<Vector<T>> EmbedImageBatch(IEnumerable<string> imagePaths)
    {
        if (imagePaths == null)
            throw new ArgumentNullException(nameof(imagePaths));

        return imagePaths.Select(path => EmbedImage(path));
    }

    private int GetImageHash(string imagePath)
    {
        // Simple hash based on file path and length
        // In production, would hash image content after preprocessing
        unchecked
        {
            int hash = 17;
            hash = (hash * 31) + imagePath.GetHashCode();

            if (File.Exists(imagePath))
            {
                var fileInfo = new FileInfo(imagePath);
                hash = (hash * 31) + fileInfo.Length.GetHashCode();
            }

            return hash;
        }
    }
}


