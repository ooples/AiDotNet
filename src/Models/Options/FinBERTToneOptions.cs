using AiDotNet.Attributes;
using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for FinBERT-tone, a BERT-base model pretrained on financial communications and fine-tuned to
/// classify the tone of analyst-report sentences.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// FinBERT-tone is BERT-base further pretrained on 4.9 billion tokens of corporate reports, earnings call transcripts
/// and analyst reports, with its own WordPiece vocabulary (FinVocab, 30,873 tokens), then fine-tuned on 10,000
/// manually labelled analyst-report sentences. Each sentence is labelled positive, negative or neutral.
/// </para>
/// <para><b>For Beginners:</b> Use this model to read the tone of what companies and analysts write. It gives each
/// sentence a positive, negative or neutral score. Load the published weights (yiyanghkust/finbert-tone) through
/// <see cref="FinancialBertEncoderOptions{T}.PretrainedModelPath"/> rather than training it from scratch.
/// </para>
/// <para>
/// <b>Reference:</b> Yang, Uy and Huang, "FinBERT: A Pretrained Language Model for Financial Communications", 2020.
/// https://arxiv.org/abs/2006.08097
/// </para>
/// </remarks>
public class FinBERTToneOptions<T> : FinancialBertEncoderOptions<T>
{
    /// <summary>
    /// Initializes a new instance with default FinBERT-Tone configuration.
    /// </summary>
    public FinBERTToneOptions() { }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    public FinBERTToneOptions(FinBERTToneOptions<T> other)
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
        NumToneClasses = other.NumToneClasses;
        DropoutRate = other.DropoutRate;
    }

    /// <summary>Maximum sequence length in tokens (default: 512).</summary>
    public int MaxSequenceLength { get; set; } = 512;

    /// <summary>Vocabulary size (default: 30873, FinVocab).</summary>
    [PaperDefault(30873, Source = "Yang, Uy and Huang 2020, Sec. 4: the uncased FinVocab has 30,873 tokens")]
    public int VocabularySize { get; set; } = 30873;

    /// <summary>Hidden dimension (default: 768 for BERT-base).</summary>
    public int HiddenDimension { get; set; } = 768;

    /// <summary>Number of attention heads (default: 12).</summary>
    public int NumAttentionHeads { get; set; } = 12;

    /// <summary>Intermediate feed-forward dimension (default: 3072).</summary>
    public int IntermediateDimension { get; set; } = 3072;

    /// <summary>Number of transformer layers (default: 12).</summary>
    public int NumLayers { get; set; } = 12;

    /// <summary>
    /// Number of tone classes (default: 3: positive, negative and neutral, as in the paper).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The paper labels each sentence positive, negative or neutral. Set another count
    /// only when fine-tuning on your own labels.
    /// </para>
    /// </remarks>
    [PaperDefault(3, Source = "Yang, Uy and Huang 2020, Sec. 5.1: each analyst-report sentence is positive, negative or neutral")]
    public int NumToneClasses { get; set; } = 3;

    /// <summary>Dropout rate (default: 0.1).</summary>
    public double DropoutRate { get; set; } = 0.1;

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
        ValidateBertEncoderOptions(MaxSequenceLength);
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
        if (NumToneClasses < 1)
            throw new ArgumentException("NumToneClasses must be at least 1.", nameof(NumToneClasses));
        if (DropoutRate < 0 || DropoutRate >= 1)
            throw new ArgumentException("DropoutRate must be between 0 (inclusive) and 1 (exclusive).", nameof(DropoutRate));
    }
}
