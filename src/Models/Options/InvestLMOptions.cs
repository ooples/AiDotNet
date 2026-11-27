using AiDotNet.Attributes;
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
public class InvestLMOptions<T> : FinancialLanguageModelOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default InvestLM configuration.
    /// </summary>
    public InvestLMOptions() { }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    public InvestLMOptions(InvestLMOptions<T> other)
        : base(other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        // Seed is declared on ModelOptions rather than in this file, so a copy constructor
        // written from the local declarations alone misses it. Losing it on a clone silently
        // changes deterministic initialization.
        Seed = other.Seed;
        MaxSequenceLength = other.MaxSequenceLength;
        VocabularySize = other.VocabularySize;
        HiddenDimension = other.HiddenDimension;
        NumAttentionHeads = other.NumAttentionHeads;
        IntermediateDimension = other.IntermediateDimension;
        NumLayers = other.NumLayers;
        NumClasses = other.NumClasses;
        DropoutRate = other.DropoutRate;
    }

    /// <summary>Maximum sequence length in tokens (default: 2048, LLaMA-65B's context).</summary>
    [PaperDefault(2048, Source = "Touvron et al. 2023 (LLaMA), Table 2 and the released LLaMA-65B config: 2k context")]
    public int MaxSequenceLength { get; set; } = 2048;

    /// <summary>Vocabulary size (default: 32000, the LLaMA SentencePiece vocabulary).</summary>
    [PaperDefault(32000, Source = "Touvron et al. 2023 (LLaMA), Table 2 and the released LLaMA-65B config")]
    public int VocabularySize { get; set; } = 32000;

    /// <summary>Hidden dimension (default: 8192, LLaMA-65B, which InvestLM fine-tunes).</summary>
    [PaperDefault(8192, Source = "Touvron et al. 2023 (LLaMA), Table 2 and the released LLaMA-65B config")]
    public int HiddenDimension { get; set; } = 8192;

    /// <summary>Number of attention heads (default: 64, LLaMA-65B).</summary>
    [PaperDefault(64, Source = "Touvron et al. 2023 (LLaMA), Table 2 and the released LLaMA-65B config")]
    public int NumAttentionHeads { get; set; } = 64;

    /// <summary>SwiGLU feed-forward width (default: 22016, LLaMA-65B).</summary>
    [PaperDefault(22016, Source = "Touvron et al. 2023 (LLaMA), Table 2 and the released LLaMA-65B config: SwiGLU width")]
    public int IntermediateDimension { get; set; } = 22016;

    /// <summary>Number of decoder blocks (default: 80, LLaMA-65B).</summary>
    [PaperDefault(80, Source = "Touvron et al. 2023 (LLaMA), Table 2 and the released LLaMA-65B config")]
    public int NumLayers { get; set; } = 80;

    /// <summary>Number of output classes (default: 3 for recommendation: buy/hold/sell).</summary>
    public int NumClasses { get; set; } = 3;

    /// <summary>Dropout rate (default: 0.0; LLaMA is trained without dropout).</summary>
    [PaperDefault(0.0, Source = "Touvron et al. 2023 (LLaMA), Table 2 and the released LLaMA-65B config: no dropout")]
    public double DropoutRate { get; set; } = 0.0;

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
        ValidateNLPOptions();
        if (MaxSequenceLength < 1)
            throw new ArgumentException("MaxSequenceLength must be at least 1.", nameof(MaxSequenceLength));
        if (VocabularySize < 1)
            throw new ArgumentException("VocabularySize must be at least 1.", nameof(VocabularySize));
        if (HiddenDimension < 1)
            throw new ArgumentException("HiddenDimension must be at least 1.", nameof(HiddenDimension));
        if (NumAttentionHeads < 1)
            throw new ArgumentException("NumAttentionHeads must be at least 1.", nameof(NumAttentionHeads));
        if (NumAttentionHeads >= 1 && HiddenDimension % NumAttentionHeads != 0)
            throw new ArgumentException("HiddenDimension must be divisible by NumAttentionHeads.", nameof(NumAttentionHeads));
        if (IntermediateDimension < 1)
            throw new ArgumentException("IntermediateDimension must be at least 1.", nameof(IntermediateDimension));
        if (NumLayers < 1)
            throw new ArgumentException("NumLayers must be at least 1.", nameof(NumLayers));
        if (NumClasses < 1)
            throw new ArgumentException("NumClasses must be at least 1.", nameof(NumClasses));
        if (DropoutRate < 0 || DropoutRate >= 1)
            throw new ArgumentException("DropoutRate must be between 0 (inclusive) and 1 (exclusive).", nameof(DropoutRate));
    }
}
