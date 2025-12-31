using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for Flamingo-style models with in-context visual learning capabilities.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Flamingo is a visual language model that excels at few-shot learning - it can learn new tasks
/// from just a few examples provided in the context. It uses gated cross-attention layers
/// interleaved with frozen LLM layers to integrate visual information.
/// </para>
/// <para><b>For Beginners:</b> Flamingo learns new visual tasks from examples you show it!
///
/// Key innovation - In-context learning:
/// - Show Flamingo a few example image-text pairs
/// - It learns the pattern from these examples
/// - Apply the pattern to new images WITHOUT any training
///
/// Architecture:
/// 1. Vision Encoder: Extracts image features (Perceiver Resampler)
/// 2. Gated Cross-Attention: Injects visual info into language model
/// 3. Frozen LLM: Chinchilla-based language model
///
/// Example use case:
/// - Show 3 examples: [image1] "A red apple" [image2] "A blue car" [image3] "A green tree"
/// - Ask about new image: [image4] "What color?"
/// - Flamingo learns from examples that you want the color, answers correctly!
///
/// Why Flamingo is revolutionary:
/// - No fine-tuning needed for new tasks
/// - Adapts to new visual concepts on-the-fly
/// - Strong performance with minimal examples
/// </para>
/// </remarks>
public interface IFlamingoModel<T> : IMultimodalEmbedding<T>
{
    /// <summary>
    /// Gets the number of visual tokens per image after the Perceiver Resampler.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Perceiver Resampler compresses visual features to a fixed number of tokens
    /// (typically 64) regardless of input image size. This enables efficient processing
    /// of multiple images in context.
    /// </para>
    /// </remarks>
    int NumPerceiverTokens { get; }

    /// <summary>
    /// Gets the maximum number of images that can be processed in a single context.
    /// </summary>
    int MaxImagesInContext { get; }

    /// <summary>
    /// Gets the language model backbone used for generation.
    /// </summary>
    /// <remarks>
    /// Flamingo typically uses <see cref="LanguageModelBackbone.Chinchilla"/> as the backbone.
    /// </remarks>
    LanguageModelBackbone LanguageModelBackbone { get; }

    /// <summary>
    /// Performs few-shot visual learning with interleaved image-text examples.
    /// </summary>
    /// <param name="examples">Few-shot examples as (image, text) pairs.</param>
    /// <param name="queryImage">The new image to process.</param>
    /// <param name="queryPrompt">Optional prompt for the query (e.g., "What is this?").</param>
    /// <param name="maxLength">Maximum tokens to generate.</param>
    /// <returns>The generated response based on learned pattern.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Learn a task from examples, then apply it!
    ///
    /// Example - Learning to identify dog breeds:
    /// Examples:
    /// - [image of labrador] "This is a Labrador Retriever"
    /// - [image of poodle] "This is a Poodle"
    /// - [image of beagle] "This is a Beagle"
    ///
    /// Query: [image of golden retriever] "This is a..."
    /// Response: "Golden Retriever"
    ///
    /// Flamingo learned the pattern from examples without any training!
    /// </para>
    /// </remarks>
    string FewShotGenerate(
        IEnumerable<(Tensor<T> Image, string Text)> examples,
        Tensor<T> queryImage,
        string? queryPrompt = null,
        int maxLength = 256);

    /// <summary>
    /// Generates text for multiple images interleaved in a single context.
    /// </summary>
    /// <param name="images">Sequence of images to process.</param>
    /// <param name="prompt">Prompt that may reference images using special tokens.</param>
    /// <param name="maxLength">Maximum tokens to generate.</param>
    /// <returns>Generated text response.</returns>
    /// <remarks>
    /// <para>
    /// Supports prompts like: "&lt;image&gt; shows a cat and &lt;image&gt; shows a dog. Compare them."
    /// where &lt;image&gt; tokens are replaced with corresponding image features.
    /// </para>
    /// </remarks>
    string GenerateWithMultipleImages(
        IEnumerable<Tensor<T>> images,
        string prompt,
        int maxLength = 512);

    /// <summary>
    /// Performs in-context visual classification without explicit labels.
    /// </summary>
    /// <param name="labeledExamples">Examples with (image, label) pairs.</param>
    /// <param name="queryImage">The image to classify.</param>
    /// <returns>Dictionary mapping labels to confidence scores.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Classify images using just a few examples!
    ///
    /// Instead of training a classifier on thousands of images:
    /// 1. Show a few examples per class
    /// 2. Flamingo learns the categories
    /// 3. It can now classify new images
    ///
    /// This is "few-shot classification" - works with any categories!
    /// </para>
    /// </remarks>
    Dictionary<string, T> InContextClassify(
        IEnumerable<(Tensor<T> Image, string Label)> labeledExamples,
        Tensor<T> queryImage);

    /// <summary>
    /// Performs visual question answering with few-shot examples.
    /// </summary>
    /// <param name="examples">Example (image, question, answer) tuples.</param>
    /// <param name="queryImage">The image to ask about.</param>
    /// <param name="question">The question to answer.</param>
    /// <returns>The generated answer.</returns>
    string FewShotVQA(
        IEnumerable<(Tensor<T> Image, string Question, string Answer)> examples,
        Tensor<T> queryImage,
        string question);

    /// <summary>
    /// Extracts visual features using the Perceiver Resampler.
    /// </summary>
    /// <param name="image">The preprocessed image tensor.</param>
    /// <returns>Resampled visual tokens with shape [numPerceiverTokens, hiddenDim].</returns>
    /// <remarks>
    /// <para>
    /// The Perceiver Resampler uses cross-attention with learnable queries to compress
    /// variable-length visual features into a fixed number of tokens.
    /// </para>
    /// </remarks>
    Tensor<T> ExtractPerceiverFeatures(Tensor<T> image);

    /// <summary>
    /// Generates captions for a video represented as a sequence of frames.
    /// </summary>
    /// <param name="frames">Sequence of video frame tensors.</param>
    /// <param name="prompt">Optional prompt to guide generation.</param>
    /// <param name="maxLength">Maximum tokens to generate.</param>
    /// <returns>Generated video description.</returns>
    /// <remarks>
    /// <para>
    /// Flamingo can process multiple frames as separate images interleaved in context,
    /// enabling basic video understanding.
    /// </para>
    /// </remarks>
    string DescribeVideo(
        IEnumerable<Tensor<T>> frames,
        string? prompt = null,
        int maxLength = 256);

    /// <summary>
    /// Computes the log probability of a given text completion for an image.
    /// </summary>
    /// <param name="image">The preprocessed image tensor.</param>
    /// <param name="text">The text to score.</param>
    /// <returns>Log probability of the text given the image.</returns>
    /// <remarks>
    /// <para>
    /// Useful for ranking candidate captions or performing discriminative tasks
    /// with a generative model.
    /// </para>
    /// </remarks>
    T ScoreImageText(Tensor<T> image, string text);

    /// <summary>
    /// Retrieves the most similar images from a database using few-shot context.
    /// </summary>
    /// <param name="queryExamples">Example images representing what you're looking for.</param>
    /// <param name="queryDescription">Optional text description of desired images.</param>
    /// <param name="candidateImages">Database of images to search.</param>
    /// <param name="topK">Number of results to return.</param>
    /// <returns>Indices of most similar images with scores.</returns>
    IEnumerable<(int Index, T Score)> FewShotImageRetrieval(
        IEnumerable<Tensor<T>> queryExamples,
        string? queryDescription,
        IEnumerable<Tensor<T>> candidateImages,
        int topK = 10);
}
