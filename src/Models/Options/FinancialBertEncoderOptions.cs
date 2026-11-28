using System;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Settings the BERT-based financial language models share (FinBERT, FinancialBERT, SEC-BERT, FinBERT-tone): the
/// encoder's normalization placement, its embeddings and attention dropout, and loading a pretrained checkpoint.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// All four models are BERT-base encoders (Devlin et al. 2019) further pretrained or fine-tuned on financial text, so
/// they share BERT's layout: token, position and segment embeddings, a stack of Post-LN transformer blocks, and a pooler
/// that passes the [CLS] position through a tanh-activated linear layer.
/// </para>
/// <para><b>For Beginners:</b> Set <see cref="PretrainedModelPath"/> to a downloaded Hugging Face checkpoint folder to
/// start from the published weights instead of random ones. That is how these models are meant to be used: training
/// one from scratch needs far more text than fine-tuning does.
/// </para>
/// </remarks>
public abstract class FinancialBertEncoderOptions<T> : FinancialLanguageModelOptions<T>
{
    /// <summary>
    /// Initializes the options with BERT-base's defaults.
    /// </summary>
    protected FinancialBertEncoderOptions()
    {
    }

    /// <summary>
    /// Initializes the options by copying the shared settings from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    protected FinancialBertEncoderOptions(FinancialBertEncoderOptions<T> other)
        : base(other)
    {
        NormPlacement = other.NormPlacement;
        AttentionDropoutRate = other.AttentionDropoutRate;
        TypeVocabSize = other.TypeVocabSize;
        MaxPositionEmbeddings = other.MaxPositionEmbeddings;
        UsePretrainedWeights = other.UsePretrainedWeights;
        PretrainedModelPath = other.PretrainedModelPath;
    }

    /// <summary>
    /// Gets or sets where each transformer block normalizes.
    /// </summary>
    /// <value><see cref="TransformerNormPlacement.PostNorm"/> (the default), BERT's own placement.</value>
    /// <remarks>
    /// <para>
    /// BERT normalizes the residual sum after each sublayer (Post-LN, Devlin et al. 2019, following Vaswani et al.
    /// 2017), and a pretrained BERT checkpoint only reproduces its outputs with that placement. Pre-LN normalizes each
    /// sublayer's input instead and trains more stably from scratch without learning-rate warmup (Xiong et al. 2020).
    /// </para>
    /// <para><b>For Beginners:</b> Keep the default when loading pretrained weights. Pre-LN can help when training a
    /// small model from random weights.
    /// </para>
    /// </remarks>
    public TransformerNormPlacement NormPlacement { get; set; } = TransformerNormPlacement.PostNorm;

    /// <summary>
    /// Gets or sets the dropout rate applied to the attention probabilities.
    /// </summary>
    /// <value>The attention dropout rate, defaulting to 0.1 (BERT's attention_probs_dropout_prob).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Randomly ignores some attention links during training so the model does not
    /// depend on any single word pairing. It has no effect at prediction time.
    /// </para>
    /// </remarks>
    public double AttentionDropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the number of segment types the segment embedding distinguishes.
    /// </summary>
    /// <value>The segment vocabulary size, defaulting to 2 (BERT's type_vocab_size: sentence A and sentence B).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> BERT can read a pair of sentences and marks each token with which one it came
    /// from. Single-sentence classification uses only the first segment.
    /// </para>
    /// </remarks>
    public int TypeVocabSize { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of positions the position embedding has entries for.
    /// </summary>
    /// <value>The position table size, defaulting to 512 (BERT's max_position_embeddings).</value>
    /// <remarks>
    /// <para>
    /// Must be at least the model's maximum sequence length. A pretrained checkpoint fixes it: its position table has
    /// this many rows.
    /// </para>
    /// <para><b>For Beginners:</b> The model learns one vector per position, so it cannot read past this many tokens.
    /// </para>
    /// </remarks>
    public int MaxPositionEmbeddings { get; set; } = 512;

    /// <summary>
    /// Gets or sets whether the model loads the checkpoint at <see cref="PretrainedModelPath"/> when it is built.
    /// </summary>
    /// <value>False (the default) to start from random weights; true to load <see cref="PretrainedModelPath"/>.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Turn this on, and set the path, to start from the published model.
    /// </para>
    /// </remarks>
    public bool UsePretrainedWeights { get; set; } = false;

    /// <summary>
    /// Gets or sets the Hugging Face checkpoint folder (or its <c>model.safetensors</c> file) to load.
    /// </summary>
    /// <value>The checkpoint path, empty by default.</value>
    /// <remarks>
    /// <para>
    /// The checkpoint must be a BERT encoder with this model's dimensions, in safetensors format (for example
    /// ProsusAI/finbert, ahmedrachid/FinancialBERT, nlpaueb/sec-bert-base or yiyanghkust/finbert-tone). The encoder and
    /// pooler weights are loaded; a classifier is loaded too when the checkpoint has one with this model's class count.
    /// When <see cref="FinancialLanguageModelOptions{T}.Tokenizer"/> is null, the checkpoint folder's tokenizer is used.
    /// </para>
    /// <para><b>For Beginners:</b> Download the model folder from Hugging Face and put its path here.
    /// </para>
    /// </remarks>
    public string PretrainedModelPath { get; set; } = string.Empty;

    /// <summary>
    /// Checks the BERT settings and the shared ones.
    /// </summary>
    /// <param name="maxSequenceLength">The model's maximum sequence length, which the position table must cover.</param>
    /// <exception cref="ArgumentException">Thrown when a setting is out of range or the settings contradict each
    /// other.</exception>
    protected void ValidateBertEncoderOptions(int maxSequenceLength)
    {
        ValidateNLPOptions();
        if (!Enum.IsDefined(typeof(TransformerNormPlacement), NormPlacement))
            throw new ArgumentException("NormPlacement must be a valid TransformerNormPlacement value.",
                nameof(NormPlacement));
        if (AttentionDropoutRate < 0 || AttentionDropoutRate >= 1)
            throw new ArgumentException("AttentionDropoutRate must be between 0 (inclusive) and 1 (exclusive).",
                nameof(AttentionDropoutRate));
        if (TypeVocabSize < 1)
            throw new ArgumentException("TypeVocabSize must be at least 1.", nameof(TypeVocabSize));
        if (MaxPositionEmbeddings < maxSequenceLength)
            throw new ArgumentException(
                $"MaxPositionEmbeddings ({MaxPositionEmbeddings}) must be at least MaxSequenceLength ({maxSequenceLength}).",
                nameof(MaxPositionEmbeddings));
        if (UsePretrainedWeights && string.IsNullOrWhiteSpace(PretrainedModelPath))
            throw new ArgumentException("UsePretrainedWeights needs PretrainedModelPath to name a checkpoint.",
                nameof(PretrainedModelPath));
    }
}
