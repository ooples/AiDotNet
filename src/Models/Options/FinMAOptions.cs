using AiDotNet.Attributes;
using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for FinMA, the financial LLaMA of the PIXIU project.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// FinMA is LLaMA (Touvron et al. 2023) fine-tuned on PIXIU's financial instruction data (FIT, 136K samples) covering
/// sentiment analysis, news headline classification, named entity recognition, question answering and stock movement
/// prediction. It keeps LLaMA's decoder unchanged: pre-RMSNorm blocks with rotary position embeddings, causal
/// self-attention and a SwiGLU feed-forward network. FinMA-7B fine-tunes LLaMA-7B, whose dimensions are the defaults
/// here; FinMA-30B fine-tunes LLaMA-30B.
/// </para>
/// <para><b>For Beginners:</b> FinMA reads financial text and, in this library, classifies it: the hidden state of the
/// last token summarizes the whole input and a linear head turns it into class scores.
/// </para>
/// <para>
/// <b>Reference:</b> Xie et al., "PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for
/// Finance", 2023. https://arxiv.org/abs/2306.05443
/// </para>
/// </remarks>
public class FinMAOptions<T> : FinancialLanguageModelOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default FinMA configuration.
    /// </summary>
    public FinMAOptions() { }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    public FinMAOptions(FinMAOptions<T> other)
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

    /// <summary>Maximum sequence length in tokens (default: 2048, LLaMA-7B's context).</summary>
    [PaperDefault(2048, Source = "Touvron et al. 2023 (LLaMA), Table 2 and the released LLaMA-7B config: 2k context")]
    public int MaxSequenceLength { get; set; } = 2048;

    /// <summary>Vocabulary size (default: 32000, the LLaMA SentencePiece vocabulary).</summary>
    [PaperDefault(32000, Source = "Touvron et al. 2023 (LLaMA), Table 2 and the released LLaMA-7B config")]
    public int VocabularySize { get; set; } = 32000;

    /// <summary>Hidden dimension (default: 4096, LLaMA-7B, which FinMA-7B fine-tunes).</summary>
    [PaperDefault(4096, Source = "Touvron et al. 2023 (LLaMA), Table 2 and the released LLaMA-7B config")]
    public int HiddenDimension { get; set; } = 4096;

    /// <summary>Number of attention heads (default: 32, LLaMA-7B).</summary>
    [PaperDefault(32, Source = "Touvron et al. 2023 (LLaMA), Table 2 and the released LLaMA-7B config")]
    public int NumAttentionHeads { get; set; } = 32;

    /// <summary>SwiGLU feed-forward width (default: 11008, LLaMA-7B).</summary>
    [PaperDefault(11008, Source = "Touvron et al. 2023 (LLaMA), Table 2 and the released LLaMA-7B config: SwiGLU width")]
    public int IntermediateDimension { get; set; } = 11008;

    /// <summary>Number of decoder blocks (default: 32, LLaMA-7B).</summary>
    [PaperDefault(32, Source = "Touvron et al. 2023 (LLaMA), Table 2 and the released LLaMA-7B config")]
    public int NumLayers { get; set; } = 32;

    /// <summary>Number of output classes (default: 3).</summary>
    public int NumClasses { get; set; } = 3;

    /// <summary>Dropout rate (default: 0.0; LLaMA is trained without dropout).</summary>
    [PaperDefault(0.0, Source = "Touvron et al. 2023 (LLaMA), Table 2 and the released LLaMA-7B config: no dropout")]
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Validates the FinMA options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This ensures the model sizes and agent counts are
    /// valid so the architecture can be created safely.
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
        if (NumAgents < 1)
            throw new ArgumentException("NumAgents must be at least 1.", nameof(NumAgents));
        if (DropoutRate < 0 || DropoutRate >= 1)
            throw new ArgumentException("DropoutRate must be between 0 (inclusive) and 1 (exclusive).", nameof(DropoutRate));
    }
}
