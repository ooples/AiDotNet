using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for FinBERT-Tone model for fine-grained financial sentiment/tone analysis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// FinBERT-Tone is a FinBERT variant specifically focused on capturing fine-grained
/// sentiment and tone in financial communications. It uses 5 classes to capture more
/// nuanced sentiment than standard 3-class models.
/// </para>
/// <para><b>For Beginners:</b> FinBERT-Tone provides nuanced sentiment analysis:
///
/// <b>The Key Insight:</b>
/// Standard sentiment models use 3 classes (negative/neutral/positive), but financial
/// communications often have more subtle tones. FinBERT-Tone uses 5 classes to capture
/// gradations like "cautiously optimistic" vs "strongly bullish".
///
/// <b>What Problems Does FinBERT-Tone Solve?</b>
/// - Earnings call tone analysis (detecting management confidence levels)
/// - Forward-looking statement sentiment classification
/// - Analyst report tone assessment
/// - Corporate communication sentiment tracking
/// - Detecting subtle changes in disclosure tone over time
///
/// <b>The 5 Tone Classes:</b>
/// 1. Very Negative: Strong pessimism, significant concerns
/// 2. Negative: Pessimistic outlook, concerns mentioned
/// 3. Neutral: Factual, no clear sentiment
/// 4. Positive: Optimistic outlook, confidence expressed
/// 5. Very Positive: Strong optimism, high confidence
///
/// <b>Key Benefits:</b>
/// - Fine-grained sentiment detection
/// - Better captures management tone nuances
/// - Useful for sentiment momentum tracking
/// - Effective for earnings call analysis
/// </para>
/// <para>
/// <b>Reference:</b> Huang et al., "FinBERT: A Pretrained Language Model for Financial Communications", 2020.
/// </para>
/// </remarks>
public class FinBERTToneOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default FinBERT-Tone configuration.
    /// </summary>
    public FinBERTToneOptions() { }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    public FinBERTToneOptions(FinBERTToneOptions<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        MaxSequenceLength = other.MaxSequenceLength;
        VocabularySize = other.VocabularySize;
        HiddenDimension = other.HiddenDimension;
        NumAttentionHeads = other.NumAttentionHeads;
        IntermediateDimension = other.IntermediateDimension;
        NumLayers = other.NumLayers;
        NumToneClasses = other.NumToneClasses;
        DropoutRate = other.DropoutRate;
        UseFinegrainedTone = other.UseFinegrainedTone;
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

    /// <summary>
    /// Number of tone classes (default: 5 for fine-grained sentiment).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> 5 classes capture more nuance:
    /// - 5: Very Negative, Negative, Neutral, Positive, Very Positive
    /// - 3: Standard Negative, Neutral, Positive
    /// </para>
    /// </remarks>
    public int NumToneClasses { get; set; } = 5;

    /// <summary>Dropout rate (default: 0.1).</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Whether to use fine-grained 5-class tone (true) or standard 3-class sentiment (false).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Fine-grained tone (5 classes) is better for capturing
    /// subtle differences in management communications. Standard sentiment (3 classes)
    /// is simpler but loses nuance.
    /// </para>
    /// </remarks>
    public bool UseFinegrainedTone { get; set; } = true;

    /// <summary>
    /// Validates the FinBERT-Tone options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This checks that the dimensions and class counts
    /// are sensible before building the model.
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
        if (NumToneClasses < 1)
            throw new ArgumentException("NumToneClasses must be at least 1.", nameof(NumToneClasses));
        if (DropoutRate < 0 || DropoutRate >= 1)
            throw new ArgumentException("DropoutRate must be between 0 (inclusive) and 1 (exclusive).", nameof(DropoutRate));
    }
}
