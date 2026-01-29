using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for InvestLM (Investment Language Model).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// InvestLM is a large language model specifically designed for investment-related
/// NLP tasks including portfolio analysis, stock recommendation, and market research.
/// </para>
/// <para><b>For Beginners:</b> InvestLM focuses on investment applications:
///
/// <b>The Key Insight:</b>
/// Investment decisions require understanding of market dynamics, company fundamentals,
/// and economic indicators. InvestLM is trained to process investment-relevant text
/// and provide insights for portfolio management.
///
/// <b>What Problems Does InvestLM Solve?</b>
/// - Stock recommendation reasoning
/// - Portfolio analysis and suggestions
/// - Market research summarization
/// - Investment thesis generation
/// - Risk factor analysis
/// - Earnings call insights extraction
///
/// <b>Key Benefits:</b>
/// - Understands investment terminology
/// - Can reason about market conditions
/// - Trained on investment research corpus
/// - Supports various investment workflows
/// </para>
/// <para>
/// <b>Reference:</b> Yang et al., "InvestLM: A Large Language Model for Investment", 2023.
/// </para>
/// </remarks>
public class InvestLMOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default InvestLM configuration.
    /// </summary>
    public InvestLMOptions() { }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    public InvestLMOptions(InvestLMOptions<T> other)
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

    /// <summary>Maximum sequence length in tokens (default: 2048).</summary>
    public int MaxSequenceLength { get; set; } = 2048;

    /// <summary>Vocabulary size (default: 32000 for LLaMA-style).</summary>
    public int VocabularySize { get; set; } = 32000;

    /// <summary>Hidden dimension (default: 768).</summary>
    public int HiddenDimension { get; set; } = 768;

    /// <summary>Number of attention heads (default: 12).</summary>
    public int NumAttentionHeads { get; set; } = 12;

    /// <summary>Intermediate feed-forward dimension (default: 3072).</summary>
    public int IntermediateDimension { get; set; } = 3072;

    /// <summary>Number of transformer layers (default: 12).</summary>
    public int NumLayers { get; set; } = 12;

    /// <summary>Number of output classes (default: 3 for recommendation: buy/hold/sell).</summary>
    public int NumClasses { get; set; } = 3;

    /// <summary>Dropout rate (default: 0.1).</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Task type: "recommendation", "sentiment", "qa", "summary" (default: "recommendation").
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Different investment NLP tasks:
    /// - recommendation: Buy/hold/sell classification
    /// - sentiment: Investment sentiment analysis
    /// - qa: Investment-related question answering
    /// - summary: Investment research summarization
    /// </para>
    /// </remarks>
    public string TaskType { get; set; } = "recommendation";

    /// <summary>
    /// Validates the InvestLM options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This ensures the configuration has positive sizes
    /// and a valid task type before use.
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
