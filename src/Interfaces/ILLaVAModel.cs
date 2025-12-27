using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for LLaVA (Large Language and Vision Assistant) models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// LLaVA connects a vision encoder (like CLIP ViT) with a large language model (like LLaMA/Vicuna)
/// through a simple projection layer, enabling visual instruction-following and conversational AI
/// about images.
/// </para>
/// <para><b>For Beginners:</b> LLaVA is like giving eyes to ChatGPT!
///
/// Architecture:
/// 1. Vision Encoder (CLIP ViT): Converts images to feature vectors
/// 2. Projection Layer: Maps visual features to LLM's text embedding space
/// 3. Large Language Model (LLaMA/Vicuna): Generates responses
///
/// Key capabilities:
/// - Visual conversations: "What's in this image?" followed by "What color is the car?"
/// - Visual reasoning: Understanding relationships, counting, spatial awareness
/// - Instruction following: "Describe this image as if you were a poet"
/// - Multi-turn dialogue: Context-aware conversations about images
///
/// Why LLaVA is popular:
/// - Simple but effective architecture
/// - Open-source and reproducible
/// - Strong performance on visual understanding benchmarks
/// - Efficient training with visual instruction tuning
/// </para>
/// </remarks>
public interface ILLaVAModel<T> : IMultimodalEmbedding<T>
{
    /// <summary>
    /// Gets the type of language model backend.
    /// </summary>
    /// <remarks>
    /// Common backends include LLaMA-2, Vicuna, Mistral, etc.
    /// </remarks>
    string LanguageModelType { get; }

    /// <summary>
    /// Gets the vision encoder type.
    /// </summary>
    /// <remarks>
    /// Typically CLIP ViT-L/14 or similar vision transformer models.
    /// </remarks>
    string VisionEncoderType { get; }

    /// <summary>
    /// Gets the maximum number of visual tokens used per image.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The number of patch tokens extracted from the vision encoder.
    /// For CLIP ViT-L/14 at 336x336, this is typically 576 tokens (24x24 patches).
    /// </para>
    /// </remarks>
    int NumVisualTokens { get; }

    /// <summary>
    /// Generates a response to a text prompt about an image.
    /// </summary>
    /// <param name="image">The preprocessed image tensor.</param>
    /// <param name="prompt">The user's question or instruction about the image.</param>
    /// <param name="maxLength">Maximum number of tokens to generate.</param>
    /// <param name="temperature">Sampling temperature (0 = deterministic, higher = more creative).</param>
    /// <param name="topP">Nucleus sampling probability threshold.</param>
    /// <returns>The generated response.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Ask any question about an image!
    ///
    /// Examples:
    /// - "What is happening in this image?" → Detailed scene description
    /// - "How many people are in the photo?" → Counting and recognition
    /// - "What emotion does the person show?" → Emotional understanding
    /// - "Write a caption for social media" → Creative generation
    /// </para>
    /// </remarks>
    string Generate(
        Tensor<T> image,
        string prompt,
        int maxLength = 512,
        double temperature = 0.7,
        double topP = 0.9);

    /// <summary>
    /// Continues a multi-turn conversation about an image.
    /// </summary>
    /// <param name="image">The preprocessed image tensor.</param>
    /// <param name="conversationHistory">Previous turns as (role, content) pairs.</param>
    /// <param name="userMessage">The new user message.</param>
    /// <param name="maxLength">Maximum tokens to generate.</param>
    /// <param name="temperature">Sampling temperature.</param>
    /// <returns>The assistant's response.</returns>
    /// <remarks>
    /// <para>
    /// Enables multi-turn visual dialogue where context is preserved across turns.
    /// </para>
    /// <para><b>For Beginners:</b> Have a conversation about an image!
    ///
    /// Example conversation:
    /// User: "What's in this image?"
    /// Assistant: "A dog playing in a park with a red ball."
    /// User: "What breed is the dog?"
    /// Assistant: "It appears to be a Golden Retriever based on its golden fur and size."
    /// User: "Is it a sunny day?"
    /// Assistant: "Yes, there are shadows indicating bright sunlight and clear skies."
    /// </para>
    /// </remarks>
    string Chat(
        Tensor<T> image,
        IEnumerable<(string Role, string Content)> conversationHistory,
        string userMessage,
        int maxLength = 512,
        double temperature = 0.7);

    /// <summary>
    /// Generates multiple diverse responses for the same prompt.
    /// </summary>
    /// <param name="image">The preprocessed image tensor.</param>
    /// <param name="prompt">The user's question or instruction.</param>
    /// <param name="numResponses">Number of different responses to generate.</param>
    /// <param name="temperature">Sampling temperature for diversity.</param>
    /// <returns>Collection of generated responses with their log probabilities.</returns>
    IEnumerable<(string Response, T Score)> GenerateMultiple(
        Tensor<T> image,
        string prompt,
        int numResponses = 5,
        double temperature = 0.9);

    /// <summary>
    /// Extracts visual features before projection to LLM space.
    /// </summary>
    /// <param name="image">The preprocessed image tensor.</param>
    /// <returns>Visual feature tensor with shape [numPatches, hiddenDim].</returns>
    /// <remarks>
    /// <para>
    /// These are the raw CLIP features before being projected to match the LLM's embedding dimension.
    /// Useful for analysis or custom processing.
    /// </para>
    /// </remarks>
    Tensor<T> ExtractVisualFeatures(Tensor<T> image);

    /// <summary>
    /// Projects visual features to the LLM's embedding space.
    /// </summary>
    /// <param name="visualFeatures">Visual features from ExtractVisualFeatures.</param>
    /// <returns>Projected features matching LLM embedding dimension.</returns>
    Tensor<T> ProjectToLanguageSpace(Tensor<T> visualFeatures);

    /// <summary>
    /// Performs visual grounding to locate objects described by text.
    /// </summary>
    /// <param name="image">The preprocessed image tensor.</param>
    /// <param name="description">Description of the object to locate.</param>
    /// <returns>Bounding box coordinates [x1, y1, x2, y2] normalized to [0, 1].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Find where something is in an image!
    ///
    /// Example:
    /// - Description: "the red car on the left"
    /// - Returns: [0.1, 0.3, 0.4, 0.7] representing the car's bounding box
    /// </para>
    /// </remarks>
    Vector<T> GroundObject(Tensor<T> image, string description);

    /// <summary>
    /// Generates a detailed description of specific regions in an image.
    /// </summary>
    /// <param name="image">The preprocessed image tensor.</param>
    /// <param name="regions">List of bounding boxes [x1, y1, x2, y2] to describe.</param>
    /// <returns>Descriptions for each region.</returns>
    IEnumerable<string> DescribeRegions(Tensor<T> image, IEnumerable<Vector<T>> regions);

    /// <summary>
    /// Compares two images and describes their differences.
    /// </summary>
    /// <param name="image1">First preprocessed image tensor.</param>
    /// <param name="image2">Second preprocessed image tensor.</param>
    /// <param name="aspectsToCompare">Optional specific aspects to compare.</param>
    /// <returns>A description of the differences between the images.</returns>
    string CompareImages(
        Tensor<T> image1,
        Tensor<T> image2,
        IEnumerable<string>? aspectsToCompare = null);
}
