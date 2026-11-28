using AiDotNet.Attributes;
using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for BloombergGPT-style financial language model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BloombergGPT-style models are large language models trained on extensive financial
/// data including Bloomberg's proprietary financial text corpus along with general text.
/// </para>
/// <para><b>For Beginners:</b> BloombergGPT represents finance-specialized LLMs:
///
/// <b>The Key Insight:</b>
/// General LLMs (GPT-3, GPT-4) lack deep financial knowledge. BloombergGPT-style models
/// are trained on massive financial corpora to understand terminology, concepts, and
/// relationships specific to finance.
///
/// <b>What Problems Does BloombergGPT Solve?</b>
/// - Financial sentiment analysis with high accuracy
/// - Named entity recognition for financial entities
/// - Financial question answering
/// - News classification and headline generation
/// - Understanding complex financial documents
///
/// <b>Architecture Highlights:</b>
/// - Large-scale decoder-only transformer
/// - Trained on mixed financial and general text
/// - Supports various financial NLP benchmarks
///
/// <b>Key Benefits:</b>
/// - Deep financial domain knowledge
/// - State-of-the-art on financial NLP tasks
/// - Understands complex financial terminology
/// - Can handle nuanced financial queries
/// </para>
/// <para>
/// <b>Reference:</b> Wu et al., "BloombergGPT: A Large Language Model for Finance", 2023.
/// </para>
/// </remarks>
public class BloombergGPTOptions<T> : FinancialLanguageModelOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default configuration.
    /// </summary>
    public BloombergGPTOptions() { }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    public BloombergGPTOptions(BloombergGPTOptions<T> other)
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

    /// <summary>Maximum sequence length in tokens (default: 2048).</summary>
    [PaperDefault(2048, Source = "Wu et al. 2023, Sec. 2.4 and Table 3: 2,048-token sequences")]
    public int MaxSequenceLength { get; set; } = 2048;

    /// <summary>Vocabulary size (default: 131072, BloombergGPT's unigram vocabulary).</summary>
    [PaperDefault(131072, Source = "Wu et al. 2023, Sec. 2.3: a 131,072-token unigram vocabulary")]
    public int VocabularySize { get; set; } = 131072;

    /// <summary>Hidden dimension (default: 7680, BloombergGPT).</summary>
    [PaperDefault(7680, Source = "Wu et al. 2023, Sec. 2.4 and Table 3")]
    public int HiddenDimension { get; set; } = 7680;

    /// <summary>Number of attention heads (default: 40, BloombergGPT).</summary>
    [PaperDefault(40, Source = "Wu et al. 2023, Sec. 2.4 and Table 3")]
    public int NumAttentionHeads { get; set; } = 40;

    /// <summary>Feed-forward width (default: 30720, four times the hidden dimension).</summary>
    [PaperDefault(30720, Source = "Wu et al. 2023, Sec. 2.4 and Table 3: feed-forward width 4 x 7,680")]
    public int IntermediateDimension { get; set; } = 30720;

    /// <summary>Number of decoder blocks (default: 70, BloombergGPT).</summary>
    [PaperDefault(70, Source = "Wu et al. 2023, Sec. 2.4 and Table 3")]
    public int NumLayers { get; set; } = 70;

    /// <summary>Number of output classes (default: 3).</summary>
    public int NumClasses { get; set; } = 3;

    /// <summary>Dropout rate (default: 0.0; BloombergGPT is trained without dropout).</summary>
    [PaperDefault(0.0, Source = "Wu et al. 2023, Sec. 2.4 and Table 3: trained without dropout")]
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Validates the BloombergGPT options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This checks that model dimensions and settings are valid
    /// before creating the network.
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
