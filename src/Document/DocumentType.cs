namespace AiDotNet.Document;

/// <summary>
/// Types of documents supported by document AI models.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Document AI models can be specialized for different document types.
/// Use these flags to specify which types a model supports or to filter processing.
/// Multiple types can be combined using the | operator.
/// </para>
/// </remarks>
[Flags]
public enum DocumentType
{
    /// <summary>
    /// No specific document type.
    /// </summary>
    None = 0,

    /// <summary>
    /// Business documents like invoices, receipts, purchase orders, and contracts.
    /// </summary>
    BusinessDocument = 1,

    /// <summary>
    /// Scientific and academic papers with figures, tables, and citations.
    /// </summary>
    ScientificPaper = 2,

    /// <summary>
    /// Structured forms with fields, checkboxes, and signatures.
    /// </summary>
    Form = 4,

    /// <summary>
    /// Business and financial reports with complex layouts.
    /// </summary>
    Report = 8,

    /// <summary>
    /// Letters and general correspondence.
    /// </summary>
    Letter = 16,

    /// <summary>
    /// Handwritten documents and notes.
    /// </summary>
    Handwritten = 32,

    /// <summary>
    /// General scanned documents.
    /// </summary>
    ScannedDocument = 64,

    /// <summary>
    /// Screenshots of web pages.
    /// </summary>
    WebPage = 128,

    /// <summary>
    /// Charts, diagrams, and infographics.
    /// </summary>
    Infographic = 256,

    /// <summary>
    /// All document types supported.
    /// </summary>
    All = ~0
}
