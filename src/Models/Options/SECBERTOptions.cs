using System;
using AiDotNet.Finance.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for SEC-BERT model specialized for SEC filings analysis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// SEC-BERT is a BERT model fine-tuned specifically on SEC filings including 10-K, 10-Q,
/// 8-K, and other regulatory documents to understand financial disclosure language.
/// </para>
/// <para><b>For Beginners:</b> SEC-BERT specializes in regulatory filing language:
///
/// <b>The Key Insight:</b>
/// SEC filings use formal legal and accounting language that differs from general financial
/// news. Terms like "material adverse effect", "going concern", and "contingent liability"
/// have specific meanings that SEC-BERT understands.
///
/// <b>What Problems Does SEC-BERT Solve?</b>
/// - Analyzing 10-K/10-Q annual and quarterly reports
/// - Processing 8-K current reports for material events
/// - Extracting risk factors from Item 1A disclosures
/// - Understanding MD&amp;A (Management Discussion and Analysis) sections
/// - Detecting changes in disclosure language over time
///
/// <b>Key Benefits:</b>
/// - Trained on millions of SEC filing documents
/// - Understands regulatory terminology and structure
/// - Captures subtle changes in disclosure tone
/// - Effective for compliance and risk assessment tasks
/// </para>
/// <para>
/// <b>Reference:</b> Loukas et al., "SEC-BERT: A Domain-Specific Language Model for SEC Filings", 2022.
/// </para>
/// </remarks>
public class SECBERTOptions<T> : ModelOptions
{
    /// <summary>
    /// Initializes a new instance with default SEC-BERT configuration.
    /// </summary>
    public SECBERTOptions() { }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    public SECBERTOptions(SECBERTOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        MaxSequenceLength = other.MaxSequenceLength;
        VocabularySize = other.VocabularySize;
        HiddenDimension = other.HiddenDimension;
        NumAttentionHeads = other.NumAttentionHeads;
        IntermediateDimension = other.IntermediateDimension;
        NumLayers = other.NumLayers;
        NumClasses = other.NumClasses;
        DropoutRate = other.DropoutRate;
        TaskType = other.TaskType;
    }

    /// <summary>Maximum sequence length in tokens (default: 512).</summary>
    public int MaxSequenceLength { get; set; } = 512;

    /// <summary>Vocabulary size (default: 30522 for BERT-base).</summary>
    public int VocabularySize { get; set; } = 30522;

    /// <summary>Hidden dimension (default: 768 for BERT-base).</summary>
    public int HiddenDimension { get; set; } = 768;

    /// <summary>Number of attention heads (default: 12).</summary>
    public int NumAttentionHeads { get; set; } = 12;

    /// <summary>Intermediate feed-forward dimension (default: 3072).</summary>
    public int IntermediateDimension { get; set; } = 3072;

    /// <summary>Number of transformer layers (default: 12).</summary>
    public int NumLayers { get; set; } = 12;

    /// <summary>Number of output classes for classification tasks (default: 2 for binary).</summary>
    public int NumClasses { get; set; } = 2;

    /// <summary>Dropout rate (default: 0.1).</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Task type for the SEC-BERT model (default: Classification).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Different tasks require different output heads:
    /// - Classification: Document-level labels (sentiment, type)
    /// - NamedEntityRecognition: Token-level entity labels
    /// - QuestionAnswering: Extract answer spans from context
    /// </para>
    /// </remarks>
    public FinancialNLPTaskType TaskType { get; set; } = FinancialNLPTaskType.Classification;

    /// <summary>
    /// Validates the SEC-BERT options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This checks that the settings are positive and sensible
    /// so the model won't fail during setup or training.
    /// </para>
    /// </remarks>
    public void Validate()
    {
        if (MaxSequenceLength < 1)
            throw new ArgumentException("MaxSequenceLength must be at least 1.", nameof(MaxSequenceLength));
        if (VocabularySize < 1)
            throw new ArgumentException("VocabularySize must be at least 1.", nameof(VocabularySize));
        if (HiddenDimension < 1)
            throw new ArgumentException("HiddenDimension must be at least 1.", nameof(HiddenDimension));
        if (NumAttentionHeads < 1)
            throw new ArgumentException("NumAttentionHeads must be at least 1.", nameof(NumAttentionHeads));
        if (IntermediateDimension < 1)
            throw new ArgumentException("IntermediateDimension must be at least 1.", nameof(IntermediateDimension));
        if (NumLayers < 1)
            throw new ArgumentException("NumLayers must be at least 1.", nameof(NumLayers));
        if (NumClasses < 1)
            throw new ArgumentException("NumClasses must be at least 1.", nameof(NumClasses));
        if (DropoutRate < 0 || DropoutRate >= 1)
            throw new ArgumentException("DropoutRate must be between 0 (inclusive) and 1 (exclusive).", nameof(DropoutRate));
        if (!Enum.IsDefined(typeof(FinancialNLPTaskType), TaskType))
            throw new ArgumentException("TaskType must be a valid FinancialNLPTaskType value.", nameof(TaskType));
    }
}
