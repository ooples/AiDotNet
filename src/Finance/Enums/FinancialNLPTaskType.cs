namespace AiDotNet.Finance.Enums;

/// <summary>
/// Defines the different types of NLP tasks for financial text processing.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Financial NLP models can perform various tasks on text data.
/// This enum specifies which task a model should be configured for.
/// </para>
/// </remarks>
public enum FinancialNLPTaskType
{
    /// <summary>
    /// Text classification task (e.g., sentiment analysis, document categorization).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Classification assigns one label to an entire document or sentence.
    /// Examples: sentiment (positive/negative/neutral), document type (10-K/10-Q/8-K).
    /// </para>
    /// </remarks>
    Classification,

    /// <summary>
    /// Named Entity Recognition (NER) task.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> NER identifies and extracts entities like company names,
    /// ticker symbols, monetary amounts, dates, and other financial entities from text.
    /// </para>
    /// </remarks>
    NamedEntityRecognition,

    /// <summary>
    /// Question Answering (QA) task.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> QA models answer questions about a given text passage.
    /// Example: "What was the company's revenue in Q3?" given an earnings report.
    /// </para>
    /// </remarks>
    QuestionAnswering,

    /// <summary>
    /// Text summarization task.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Summarization condenses long documents into shorter summaries.
    /// Useful for summarizing lengthy SEC filings or earnings call transcripts.
    /// </para>
    /// </remarks>
    Summarization,

    /// <summary>
    /// Relationship extraction task.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Identifies relationships between entities in text.
    /// Example: "Apple acquired Beats" -> (Apple, acquired, Beats).
    /// </para>
    /// </remarks>
    RelationshipExtraction,

    /// <summary>
    /// Sequence-to-sequence generation task.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Generates text output from text input.
    /// Used for tasks like paraphrasing or converting formats.
    /// </para>
    /// </remarks>
    SequenceToSequence,

    /// <summary>
    /// Token-level classification (e.g., POS tagging).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Assigns a label to each token in the input.
    /// Similar to NER but more general.
    /// </para>
    /// </remarks>
    TokenClassification
}
