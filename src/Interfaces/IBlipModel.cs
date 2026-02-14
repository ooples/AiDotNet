using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for BLIP (Bootstrapped Language-Image Pre-training) models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BLIP extends CLIP's capabilities with additional vision-language tasks:
/// image captioning, image-text matching, and visual question answering.
/// This interface extends <see cref="IMultimodalEmbedding{T}"/> with these features.
/// </para>
/// <para><b>For Beginners:</b> BLIP is like CLIP but with extra superpowers!
///
/// What CLIP can do:
/// - Compare images and text (are they related?)
/// - Zero-shot classification (classify without training)
///
/// What BLIP adds:
/// - Generate captions for images (describe what you see)
/// - Answer questions about images (VQA)
/// - Better image-text matching with cross-attention
///
/// BLIP was trained on a larger, cleaner dataset using a special "bootstrapping"
/// technique that improves the quality of training data automatically.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("BlipModel")]
public interface IBlipModel<T> : IMultimodalEmbedding<T>
{
    /// <summary>
    /// Generates a caption describing the content of an image.
    /// </summary>
    /// <param name="image">The preprocessed image tensor with shape [channels, height, width].</param>
    /// <param name="maxLength">Maximum number of tokens to generate. Default is 30.</param>
    /// <param name="numBeams">Number of beams for beam search. Default is 3 for quality/speed balance.</param>
    /// <returns>A generated caption describing the image.</returns>
    /// <remarks>
    /// <para>
    /// Uses the image-grounded text decoder to generate descriptive captions.
    /// The generation uses beam search by default for higher quality outputs.
    /// </para>
    /// <para><b>For Beginners:</b> This automatically describes what's in an image!
    ///
    /// Example:
    /// - Input: Photo of a dog playing fetch in a park
    /// - Output: "a brown dog catching a frisbee on a grassy field"
    ///
    /// Parameters:
    /// - maxLength: How long the caption can be (30 = roughly 25 words)
    /// - numBeams: More beams = better captions but slower (3 is a good balance)
    ///
    /// Uses "beam search" - it explores multiple possible captions and picks the best one.
    /// </para>
    /// </remarks>
    string GenerateCaption(Tensor<T> image, int maxLength = 30, int numBeams = 3);

    /// <summary>
    /// Generates multiple candidate captions for an image.
    /// </summary>
    /// <param name="image">The preprocessed image tensor.</param>
    /// <param name="numCaptions">Number of captions to generate.</param>
    /// <param name="maxLength">Maximum length per caption.</param>
    /// <returns>A collection of candidate captions.</returns>
    /// <remarks>
    /// <para>
    /// Uses nucleus (top-p) sampling to generate diverse captions.
    /// Useful for getting multiple perspectives on an image's content.
    /// </para>
    /// </remarks>
    IEnumerable<string> GenerateCaptions(Tensor<T> image, int numCaptions = 5, int maxLength = 30);

    /// <summary>
    /// Determines whether a given text accurately describes an image.
    /// </summary>
    /// <param name="image">The preprocessed image tensor.</param>
    /// <param name="text">The text description to evaluate.</param>
    /// <returns>A probability score between 0 and 1 indicating match quality.</returns>
    /// <remarks>
    /// <para>
    /// Uses the Image-Text Matching (ITM) head with cross-attention between
    /// image patches and text tokens for fine-grained matching.
    /// This is more accurate than simple embedding similarity for detailed matching.
    /// </para>
    /// <para><b>For Beginners:</b> This checks if a caption accurately describes an image.
    ///
    /// Unlike simple similarity (dot product), this uses "cross-attention" which:
    /// - Looks at specific parts of the image
    /// - Compares them to specific words in the text
    /// - Gives a more accurate yes/no answer
    ///
    /// Example:
    /// - Image: A red car parked on a street
    /// - "A red vehicle on pavement" → 0.92 (accurate!)
    /// - "A blue car in a garage" → 0.15 (wrong color and location)
    ///
    /// Use this when you need precise matching, not just "related content."
    /// </para>
    /// </remarks>
    T ComputeImageTextMatch(Tensor<T> image, string text);

    /// <summary>
    /// Answers a question about an image's content.
    /// </summary>
    /// <param name="image">The preprocessed image tensor.</param>
    /// <param name="question">The question to answer (e.g., "What color is the car?").</param>
    /// <param name="maxLength">Maximum length of the answer.</param>
    /// <returns>The generated answer.</returns>
    /// <remarks>
    /// <para>
    /// Visual Question Answering (VQA) generates natural language answers to questions
    /// about image content. The model uses cross-attention to focus on relevant image
    /// regions when generating the answer.
    /// </para>
    /// <para><b>For Beginners:</b> Ask questions about images and get answers!
    ///
    /// Examples:
    /// - Image: Photo of a kitchen
    /// - "What appliances are visible?" → "refrigerator, microwave, and stove"
    /// - "What color are the cabinets?" → "white"
    /// - "Is there a window?" → "yes, above the sink"
    ///
    /// This is useful for:
    /// - Accessibility (describe images for visually impaired users)
    /// - Content moderation (is there alcohol in this photo?)
    /// - Data extraction (what brand is this product?)
    /// </para>
    /// </remarks>
    string AnswerQuestion(Tensor<T> image, string question, int maxLength = 20);

    /// <summary>
    /// Ranks a set of candidate captions by how well they match an image.
    /// </summary>
    /// <param name="image">The preprocessed image tensor.</param>
    /// <param name="candidates">The candidate captions to rank.</param>
    /// <returns>Captions ranked by match score, from best to worst.</returns>
    /// <remarks>
    /// <para>
    /// Uses the ITM head to score each candidate, then returns them in descending order.
    /// Useful for caption reranking in retrieval applications.
    /// </para>
    /// </remarks>
    IEnumerable<(string Caption, T Score)> RankCaptions(Tensor<T> image, IEnumerable<string> candidates);

    /// <summary>
    /// Retrieves the most relevant images for a text query from a collection.
    /// </summary>
    /// <param name="query">The text query describing desired images.</param>
    /// <param name="imageEmbeddings">Pre-computed image embeddings.</param>
    /// <param name="topK">Number of results to return.</param>
    /// <returns>Indices of the top-K matching images with their scores.</returns>
    /// <remarks>
    /// <para>
    /// Performs efficient text-to-image retrieval using embedding similarity.
    /// For large collections, pre-compute and cache image embeddings.
    /// </para>
    /// </remarks>
    IEnumerable<(int Index, T Score)> RetrieveImages(
        string query,
        IEnumerable<Vector<T>> imageEmbeddings,
        int topK = 10);

    /// <summary>
    /// Retrieves the most relevant texts for an image from a collection.
    /// </summary>
    /// <param name="image">The preprocessed image tensor.</param>
    /// <param name="textEmbeddings">Pre-computed text embeddings.</param>
    /// <param name="topK">Number of results to return.</param>
    /// <returns>Indices of the top-K matching texts with their scores.</returns>
    /// <remarks>
    /// <para>
    /// Performs efficient image-to-text retrieval using embedding similarity.
    /// Useful for finding relevant captions or descriptions for images.
    /// </para>
    /// </remarks>
    IEnumerable<(int Index, T Score)> RetrieveTexts(
        Tensor<T> image,
        IEnumerable<Vector<T>> textEmbeddings,
        int topK = 10);
}
