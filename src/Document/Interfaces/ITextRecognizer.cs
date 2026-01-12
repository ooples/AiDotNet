namespace AiDotNet.Document.Interfaces;

/// <summary>
/// Interface for text recognition models that read text from cropped image regions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Text recognition models convert cropped images of text into character sequences.
/// They work on pre-detected text regions (from a text detector).
/// </para>
/// <para>
/// <b>For Beginners:</b> Text recognition is the second step in reading text from images.
/// Given a small image containing only text (like a single word or line), the recognizer
/// outputs the actual characters. This is like reading what's written in a highlighted region.
///
/// Example usage:
/// <code>
/// var recognizer = new TrOCR&lt;float&gt;(architecture);
/// var result = recognizer.RecognizeText(croppedTextImage);
/// Console.WriteLine($"Recognized: {result.Text} (confidence: {result.Confidence})");
/// </code>
/// </para>
/// </remarks>
public interface ITextRecognizer<T> : IDocumentModel<T>
{
    /// <summary>
    /// Recognizes text from a cropped image region.
    /// </summary>
    /// <param name="croppedImage">Cropped image containing text (from text detector).</param>
    /// <returns>Recognition result with text and confidence.</returns>
    TextRecognitionResult<T> RecognizeText(Tensor<T> croppedImage);

    /// <summary>
    /// Recognizes text from multiple cropped image regions (batch processing).
    /// </summary>
    /// <param name="croppedImages">List of cropped images containing text.</param>
    /// <returns>List of recognition results.</returns>
    IEnumerable<TextRecognitionResult<T>> RecognizeTextBatch(IEnumerable<Tensor<T>> croppedImages);

    /// <summary>
    /// Gets the character-level probabilities for the last recognition.
    /// </summary>
    /// <returns>Tensor of shape [sequence_length, vocab_size] with probabilities.</returns>
    Tensor<T> GetCharacterProbabilities();

    /// <summary>
    /// Gets the supported character set (alphabet) for this recognizer.
    /// </summary>
    string SupportedCharacters { get; }

    /// <summary>
    /// Gets the maximum sequence length this recognizer can output.
    /// </summary>
    new int MaxSequenceLength { get; }

    /// <summary>
    /// Gets whether this recognizer supports attention visualization.
    /// </summary>
    bool SupportsAttentionVisualization { get; }

    /// <summary>
    /// Gets the attention weights for visualization (if supported).
    /// </summary>
    /// <returns>Attention tensor showing which image regions influenced each character.</returns>
    Tensor<T>? GetAttentionWeights();
}
