using AiDotNet.Attributes;
using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for FinGPT (Financial GPT) model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FinGPT is an open-source financial large language model designed for various
/// financial NLP tasks including sentiment analysis, question answering, and text generation.
/// </para>
/// <para><b>For Beginners:</b> FinGPT brings GPT capabilities to finance:
///
/// <b>The Key Insight:</b>
/// While BERT models excel at understanding tasks (classification, NER), GPT models
/// are better at generation tasks and can handle more open-ended financial queries.
/// FinGPT adapts the GPT architecture for financial applications.
///
/// <b>What Problems Does FinGPT Solve?</b>
/// - Financial sentiment analysis with explanations
/// - Question answering about financial documents
/// - Generating financial summaries
/// - Extracting insights from earnings calls
/// - Financial text completion and generation
///
/// <b>Key Benefits:</b>
/// - Open-source and fine-tunable
/// - Supports both understanding and generation
/// - Can provide reasoning for predictions
/// - Handles longer context than BERT models
/// </para>
/// <para>
/// <b>Reference:</b> Yang et al., "FinGPT: Open-Source Financial Large Language Models", 2023.
/// </para>
/// </remarks>
public class FinGPTOptions<T> : FinancialLanguageModelOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default FinGPT configuration.
    /// </summary>
    public FinGPTOptions() { }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    public FinGPTOptions(FinGPTOptions<T> other)
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

    /// <summary>Maximum sequence length in tokens (default: 4096, Llama-2-7B's context).</summary>
    [PaperDefault(4096, Source = "Touvron et al. 2023 (Llama 2), Table 1 and the released Llama-2-7b config: 4k context")]
    public int MaxSequenceLength { get; set; } = 4096;

    /// <summary>Vocabulary size (default: 32000, the Llama-2 SentencePiece vocabulary).</summary>
    [PaperDefault(32000, Source = "Touvron et al. 2023 (Llama 2), Table 1 and the released Llama-2-7b config")]
    public int VocabularySize { get; set; } = 32000;

    /// <summary>Hidden dimension (default: 4096, Llama-2-7B).</summary>
    [PaperDefault(4096, Source = "Touvron et al. 2023 (Llama 2), Table 1 and the released Llama-2-7b config")]
    public int HiddenDimension { get; set; } = 4096;

    /// <summary>Number of attention heads (default: 32, Llama-2-7B).</summary>
    [PaperDefault(32, Source = "Touvron et al. 2023 (Llama 2), Table 1 and the released Llama-2-7b config")]
    public int NumAttentionHeads { get; set; } = 32;

    /// <summary>SwiGLU feed-forward width (default: 11008, Llama-2-7B).</summary>
    [PaperDefault(11008, Source = "Touvron et al. 2023 (Llama 2), Table 1 and the released Llama-2-7b config: SwiGLU width")]
    public int IntermediateDimension { get; set; } = 11008;

    /// <summary>Number of decoder blocks (default: 32, Llama-2-7B).</summary>
    [PaperDefault(32, Source = "Touvron et al. 2023 (Llama 2), Table 1 and the released Llama-2-7b config")]
    public int NumLayers { get; set; } = 32;

    /// <summary>Number of output classes for classification tasks (default: 3).</summary>
    public int NumClasses { get; set; } = 3;

    /// <summary>Dropout rate (default: 0.0; Llama 2 is trained without dropout).</summary>
    [PaperDefault(0.0, Source = "Touvron et al. 2023 (Llama 2), Table 1 and the released Llama-2-7b config: no dropout")]
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Validates the FinGPT options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This makes sure sizes and settings are valid
    /// so the model can be built safely.
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
