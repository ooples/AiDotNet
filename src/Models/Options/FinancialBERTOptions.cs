using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for FinancialBERT model - a domain-adapted BERT for comprehensive financial analysis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// FinancialBERT is a BERT model adapted for broad financial domain understanding,
/// including news analysis, market commentary, analyst reports, and general financial text.
/// </para>
/// <para><b>For Beginners:</b> FinancialBERT provides comprehensive financial NLP:
///
/// <b>The Key Insight:</b>
/// While FinBERT focuses on sentiment and SEC-BERT on regulatory filings, FinancialBERT
/// provides broader coverage of the financial domain including analyst reports, market
/// commentary, corporate communications, and financial news.
///
/// <b>What Problems Does FinancialBERT Solve?</b>
/// - Multi-task financial text classification
/// - Extracting insights from analyst reports
/// - Processing corporate press releases
/// - Analyzing market commentary and research notes
/// - Understanding financial terminology across contexts
///
/// <b>Key Benefits:</b>
/// - Broader domain coverage than specialized models
/// - Multi-task learning capability
/// - Effective for general financial NLP applications
/// - Good baseline for financial text understanding
/// </para>
/// <para>
/// <b>Reference:</b> Huang et al., "FinancialBERT: A Pre-trained Language Model for Financial Text Mining", 2023.
/// </para>
/// </remarks>
public class FinancialBERTOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default FinancialBERT configuration.
    /// </summary>
    public FinancialBERTOptions() { }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    public FinancialBERTOptions(FinancialBERTOptions<T> other)
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
        UseMultiTask = other.UseMultiTask;
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

    /// <summary>Number of output classes (default: 3 for sentiment).</summary>
    public int NumClasses { get; set; } = 3;

    /// <summary>Dropout rate (default: 0.1).</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Task type: "sentiment", "topic", "entity", "multi" (default: "sentiment").
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Different financial NLP tasks:
    /// - sentiment: Positive/negative/neutral classification
    /// - topic: Categorize by financial topic (M&amp;A, earnings, etc.)
    /// - entity: Named entity recognition for financial entities
    /// - multi: Multi-task learning across multiple objectives
    /// </para>
    /// </remarks>
    public string TaskType { get; set; } = "sentiment";

    /// <summary>
    /// Whether to use multi-task learning (default: false).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Multi-task learning trains the model on multiple
    /// objectives simultaneously (sentiment + topic + entity), which can improve
    /// overall performance through shared representations.
    /// </para>
    /// </remarks>
    public bool UseMultiTask { get; set; } = false;

    /// <summary>
    /// Validates the FinancialBERT options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This checks that the configuration values are valid
    /// before you build or train the model.
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
        if (string.IsNullOrWhiteSpace(TaskType))
            throw new ArgumentException("TaskType cannot be null or empty.", nameof(TaskType));
    }
}
