using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Defines the interface for text encoding models that convert text into numerical representations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A text encoder transforms human-readable text into numerical representations (embeddings) that
    /// machine learning models can process. These encoders are crucial components in natural language
    /// processing tasks and multimodal models that work with both text and other data types.
    /// </para>
    /// <para><b>For Beginners:</b> Think of a text encoder as a translator between human language and computer language.
    /// 
    /// Computers can't directly understand words like "cat" or "happy". A text encoder converts these words
    /// into numbers that capture their meaning. It's like creating a unique numerical fingerprint for text:
    /// 
    /// - "cat" might become [0.2, -0.5, 0.8, ...]
    /// - "dog" might become [0.3, -0.4, 0.7, ...]
    /// - Similar words get similar numbers
    /// 
    /// This is essential for:
    /// - Text-to-image generation (like DALL-E or Stable Diffusion)
    /// - Language translation
    /// - Sentiment analysis
    /// - Question answering systems
    /// - Any task where computers need to understand text meaning
    /// 
    /// Modern text encoders like CLIP or BERT can understand context, so "bank" in "river bank"
    /// gets different numbers than "bank" in "money bank".
    /// </para>
    /// </remarks>
    public interface ITextEncoder
    {
        /// <summary>
        /// Encodes text into a numerical representation.
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <returns>A tensor containing the encoded representation of the text.</returns>
        /// <remarks>
        /// <para>
        /// This method converts natural language text into a dense numerical representation that captures
        /// semantic meaning. The encoding process typically involves tokenization, embedding lookup, and
        /// contextual processing through transformer or similar architectures.
        /// </para>
        /// <para><b>For Beginners:</b> This method turns words into numbers that computers can understand.
        /// 
        /// When you input text like "a beautiful sunset over the ocean", the encoder:
        /// 1. Breaks it into tokens (words or parts of words)
        /// 2. Converts each token to initial numbers
        /// 3. Processes these numbers to understand context and meaning
        /// 4. Outputs a final set of numbers that represent the entire text's meaning
        /// 
        /// The output tensor might have dimensions like [1, 768], meaning 768 numbers that capture
        /// all the meaning of your input text. These numbers can then be used by other models to:
        /// - Generate images matching the text
        /// - Find similar texts
        /// - Answer questions about the text
        /// - Translate to other languages
        /// 
        /// The quality of encoding determines how well AI systems understand your text input.
        /// </para>
        /// </remarks>
        Tensor<double> Encode(string text);
    }
}