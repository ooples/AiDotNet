namespace AiDotNet.Document.Interfaces;

/// <summary>
/// Interface for document question answering models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Document QA models answer natural language questions about document content,
/// combining visual understanding with text comprehension.
/// </para>
/// <para>
/// <b>For Beginners:</b> Document QA is like having a smart assistant that can read
/// a document and answer your questions about it. You show it a document image and
/// ask questions like "What is the total amount?" or "Who signed this contract?"
///
/// Example usage:
/// <code>
/// var result = documentQA.AnswerQuestion(invoiceImage, "What is the invoice number?");
/// Console.WriteLine($"Answer: {result.Answer} (confidence: {result.Confidence:P0})");
/// </code>
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("DocumentQA")]
public interface IDocumentQA<T> : IDocumentModel<T>
{
    /// <summary>
    /// Answers a question about a document.
    /// </summary>
    /// <param name="documentImage">The document image tensor.</param>
    /// <param name="question">The question to answer in natural language.</param>
    /// <returns>The answer with confidence and evidence information.</returns>
    DocumentQAResult<T> AnswerQuestion(Tensor<T> documentImage, string question);

    /// <summary>
    /// Answers a question with generation parameters.
    /// </summary>
    /// <param name="documentImage">The document image tensor.</param>
    /// <param name="question">The question to answer.</param>
    /// <param name="maxAnswerLength">Maximum length of the generated answer.</param>
    /// <param name="temperature">Sampling temperature for generation (0 = deterministic).</param>
    /// <returns>The answer result.</returns>
    DocumentQAResult<T> AnswerQuestion(Tensor<T> documentImage, string question, int maxAnswerLength, double temperature = 0.0);

    /// <summary>
    /// Answers multiple questions about a document in a batch.
    /// </summary>
    /// <param name="documentImage">The document image tensor.</param>
    /// <param name="questions">The questions to answer.</param>
    /// <returns>Answers for each question in order.</returns>
    /// <remarks>
    /// <para>
    /// Batching multiple questions is more efficient than calling AnswerQuestion
    /// repeatedly because the document encoding can be reused.
    /// </para>
    /// </remarks>
    IEnumerable<DocumentQAResult<T>> AnswerQuestions(Tensor<T> documentImage, IEnumerable<string> questions);

    /// <summary>
    /// Extracts specific fields from a document using natural language prompts.
    /// </summary>
    /// <param name="documentImage">The document image tensor.</param>
    /// <param name="fieldPrompts">Field names or extraction prompts (e.g., "invoice_number", "total_amount").</param>
    /// <returns>Dictionary mapping field names to their extracted values and confidence.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a convenient way to extract multiple pieces of information
    /// at once. Instead of asking separate questions, you provide a list of field names
    /// and the model extracts all of them from the document.
    /// </para>
    /// </remarks>
    Dictionary<string, DocumentQAResult<T>> ExtractFields(Tensor<T> documentImage, IEnumerable<string> fieldPrompts);
}
