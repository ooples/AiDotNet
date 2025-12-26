using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for multimodal embedding models that project different modalities into a shared vector space.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Multimodal embeddings enable cross-modal similarity search by mapping different data types
/// (text, images, audio) into the same high-dimensional space where similar concepts are close together.
/// This interface is designed for CLIP-style models that can understand both text and images.
/// </para>
/// <para><b>For Beginners:</b> Think of this like a universal translator between languages.
///
/// Normally:
/// - Text "a cat" and an image of a cat are completely different formats
/// - You can't directly compare bytes of text with pixels of an image
/// - They live in different "worlds"
///
/// With multimodal embeddings:
/// - Both get converted to the same "language" (a vector of numbers)
/// - Now you can measure how similar they are using simple math
/// - "a cat" text and cat.jpg image will have similar vectors
/// - "a dog" text and cat.jpg image will have different vectors
///
/// This enables amazing applications:
/// - Image search using text: Type "sunset over mountains" to find matching photos
/// - Image captioning: Find text that best matches an image
/// - Zero-shot classification: Classify images without training on those specific categories
/// - Content recommendation: Find related content across text and images
/// </para>
/// </remarks>
public interface IMultimodalEmbedding<T>
{
    /// <summary>
    /// Gets the dimensionality of the embedding vectors produced by this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Both text and image embeddings will have this same dimension, enabling direct comparison.
    /// Common dimensions are 512 (CLIP ViT-B/32) or 768 (CLIP ViT-L/14).
    /// </para>
    /// <para><b>For Beginners:</b> This is how many numbers represent each image or text.
    ///
    /// The key insight is that BOTH text and images become vectors of the SAME size.
    /// This is what allows comparison - you can't compare apples and oranges,
    /// but you can compare two lists of 512 numbers!
    /// </para>
    /// </remarks>
    int EmbeddingDimension { get; }

    /// <summary>
    /// Gets the maximum number of tokens the text encoder can process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// CLIP models typically have a maximum sequence length of 77 tokens.
    /// Text longer than this will be truncated.
    /// </para>
    /// <para><b>For Beginners:</b> This is the maximum "length" of text you can encode.
    ///
    /// A token is roughly a word or word piece. For CLIP:
    /// - Max 77 tokens means roughly 50-60 words
    /// - Longer text gets cut off at the end
    /// - This is enough for captions and short descriptions
    /// </para>
    /// </remarks>
    int MaxSequenceLength { get; }

    /// <summary>
    /// Gets the expected image size (height and width) for the vision encoder.
    /// </summary>
    /// <remarks>
    /// <para>
    /// CLIP models expect square images of a specific size (e.g., 224x224 or 336x336).
    /// Images will be resized to this dimension during preprocessing.
    /// </para>
    /// <para><b>For Beginners:</b> This is the image size the model expects.
    ///
    /// All images get resized to this square size before processing:
    /// - A 1920x1080 photo becomes 224x224
    /// - A tiny 50x50 thumbnail becomes 224x224
    /// - Aspect ratio may be adjusted (images get squished/stretched)
    /// </para>
    /// </remarks>
    int ImageSize { get; }

    /// <summary>
    /// Converts a single text string into an embedding vector.
    /// </summary>
    /// <param name="text">The text to embed (e.g., "a photo of a cat").</param>
    /// <returns>A normalized embedding vector of size <see cref="EmbeddingDimension"/>.</returns>
    /// <remarks>
    /// <para>
    /// The text is tokenized using a BPE tokenizer, processed through the text encoder,
    /// and L2-normalized to unit length. Normalization ensures that cosine similarity
    /// equals dot product, simplifying similarity calculations.
    /// </para>
    /// <para><b>For Beginners:</b> This converts text into a list of numbers that capture its meaning.
    ///
    /// Example:
    /// - Input: "a photo of a golden retriever playing fetch"
    /// - Output: Vector like [0.12, -0.45, 0.78, ..., 0.33] with 512 or 768 numbers
    ///
    /// The numbers are "normalized" (scaled) so the vector has length 1.0.
    /// This makes comparing different embeddings easier.
    /// </para>
    /// </remarks>
    Vector<T> GetTextEmbedding(string text);

    /// <summary>
    /// Converts multiple text strings into embedding vectors in a batch operation.
    /// </summary>
    /// <param name="texts">The texts to embed.</param>
    /// <returns>A collection of normalized embedding vectors.</returns>
    /// <remarks>
    /// <para>
    /// Batch processing is more efficient than processing texts individually,
    /// especially when using GPU acceleration.
    /// </para>
    /// <para><b>For Beginners:</b> Same as GetTextEmbedding, but for many texts at once.
    ///
    /// This is MUCH faster when you have multiple texts to embed:
    /// - Bad: Call GetTextEmbedding 1000 times (slow, lots of overhead)
    /// - Good: Call GetTextEmbeddings once with 1000 texts (fast, batched)
    /// </para>
    /// </remarks>
    IEnumerable<Vector<T>> GetTextEmbeddings(IEnumerable<string> texts);

    /// <summary>
    /// Converts a single image into an embedding vector.
    /// </summary>
    /// <param name="image">The preprocessed image tensor with shape [channels, height, width].</param>
    /// <returns>A normalized embedding vector of size <see cref="EmbeddingDimension"/>.</returns>
    /// <remarks>
    /// <para>
    /// The image should be preprocessed (resized, normalized) before calling this method.
    /// The tensor is processed through the vision encoder and L2-normalized.
    /// Expected shape is [3, ImageSize, ImageSize] with RGB channels in range [-1, 1] or [0, 1]
    /// depending on the specific CLIP model variant.
    /// </para>
    /// <para><b>For Beginners:</b> This converts an image into a list of numbers.
    ///
    /// Before calling this:
    /// 1. Resize your image to ImageSize x ImageSize pixels
    /// 2. Convert RGB values from 0-255 to the model's expected range
    /// 3. Arrange as [channels, height, width] tensor
    ///
    /// After calling this:
    /// - You get a vector with the same size as text embeddings
    /// - You can now compare this image to any text or other image!
    /// </para>
    /// </remarks>
    Vector<T> GetImageEmbedding(Tensor<T> image);

    /// <summary>
    /// Converts multiple images into embedding vectors in a batch operation.
    /// </summary>
    /// <param name="images">The preprocessed image tensors.</param>
    /// <returns>A collection of normalized embedding vectors.</returns>
    /// <remarks>
    /// <para>
    /// Batch processing is significantly more efficient, especially on GPU.
    /// All images must be preprocessed to the same size.
    /// </para>
    /// </remarks>
    IEnumerable<Vector<T>> GetImageEmbeddings(IEnumerable<Tensor<T>> images);

    /// <summary>
    /// Computes the similarity between a text embedding and an image embedding.
    /// </summary>
    /// <param name="textEmbedding">The text embedding vector.</param>
    /// <param name="imageEmbedding">The image embedding vector.</param>
    /// <returns>A similarity score, typically in the range [-1, 1] for normalized vectors.</returns>
    /// <remarks>
    /// <para>
    /// For normalized embeddings, this is the cosine similarity (dot product).
    /// Higher values indicate greater semantic similarity between the text and image.
    /// Typical values:
    /// - > 0.3: Strong match (text accurately describes the image)
    /// - 0.2 - 0.3: Moderate match (related content)
    /// - &lt; 0.2: Weak or no match
    /// </para>
    /// <para><b>For Beginners:</b> This measures how well the text describes the image.
    ///
    /// Example scores:
    /// - "a golden retriever" + photo of a golden retriever → ~0.35 (great match!)
    /// - "a dog" + photo of a golden retriever → ~0.28 (good match)
    /// - "a car" + photo of a golden retriever → ~0.12 (poor match)
    /// - "abstract art" + photo of a golden retriever → ~0.05 (no match)
    ///
    /// The score is based on how close the two vectors are in the embedding space.
    /// </para>
    /// </remarks>
    T ComputeSimilarity(Vector<T> textEmbedding, Vector<T> imageEmbedding);

    /// <summary>
    /// Performs zero-shot image classification by comparing an image to a set of text labels.
    /// </summary>
    /// <param name="image">The preprocessed image tensor to classify.</param>
    /// <param name="classLabels">The candidate class labels (e.g., ["cat", "dog", "bird"]).</param>
    /// <returns>A dictionary mapping each label to its probability score (sums to 1.0).</returns>
    /// <remarks>
    /// <para>
    /// This method enables classification without training on specific categories.
    /// The image is compared to each text label, and softmax is applied to convert
    /// similarities into a probability distribution.
    /// </para>
    /// <para><b>For Beginners:</b> This lets you classify images WITHOUT training!
    ///
    /// Traditional image classification:
    /// 1. Collect 1000s of labeled images per category
    /// 2. Train a model for weeks
    /// 3. Can only recognize the categories you trained on
    ///
    /// Zero-shot classification with CLIP:
    /// 1. Just provide category names as text
    /// 2. CLIP compares the image to each category
    /// 3. Works with ANY categories you can describe in text!
    ///
    /// Example:
    /// - Image: A photo of a beagle
    /// - Labels: ["cat", "dog", "bird", "fish"]
    /// - Result: {"cat": 0.05, "dog": 0.85, "bird": 0.07, "fish": 0.03}
    ///
    /// The model "knows" it's a dog without ever being trained on dog images!
    /// </para>
    /// </remarks>
    Dictionary<string, T> ZeroShotClassify(Tensor<T> image, IEnumerable<string> classLabels);
}
