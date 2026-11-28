using System;
using System.Collections.Generic;
using AiDotNet.Finance.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Models.Options;

/// <summary>
/// Settings every financial language model shares: what it predicts, how its head is built, how text becomes token
/// ids, and which of its layers fine-tuning trains.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// A financial language model is an <b>encoder</b> followed by a <b>head</b>. The encoder turns token ids into hidden
/// states and, for sequence classification, pools them into one vector per input (the [CLS] position for a BERT model,
/// the last position for a causal decoder). The head turns that vector into class scores.
/// <c>GetSequenceEmbedding</c> returns the encoder's pooled output; <c>Predict</c> runs the encoder and then the head.
/// </para>
/// <para>
/// <b>Custom layers.</b> Layers passed through the architecture replace the encoder. The head is then the default head,
/// the layers <see cref="ClassificationHeadFactory"/> builds, or nothing when
/// <see cref="UseDefaultClassificationHead"/> is false and no factory is set, in which case the architecture's layers
/// are the whole model.
/// </para>
/// <para><b>For Beginners:</b> You normally leave all of these alone. Set <see cref="Tokenizer"/> to analyse raw text
/// with a vocabulary of your own, <see cref="FreezeBaseModel"/> or <see cref="NumFineTuneLayers"/> to fine-tune only
/// the top of a pretrained model, and <see cref="ClassificationHeadFactory"/> to put your own layers on top of the
/// encoder.
/// </para>
/// </remarks>
public abstract class FinancialLanguageModelOptions<T> : ModelOptions
{
    /// <summary>
    /// Initializes the options with their defaults.
    /// </summary>
    protected FinancialLanguageModelOptions()
    {
    }

    /// <summary>
    /// Initializes the options by copying the shared settings from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    protected FinancialLanguageModelOptions(FinancialLanguageModelOptions<T> other)
    {
        if (other is null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        TaskType = other.TaskType;
        ClassificationHeadFactory = other.ClassificationHeadFactory;
        UseDefaultClassificationHead = other.UseDefaultClassificationHead;
        Tokenizer = other.Tokenizer;
        FreezeBaseModel = other.FreezeBaseModel;
        NumFineTuneLayers = other.NumFineTuneLayers;
    }

    /// <summary>
    /// Gets or sets what the model predicts.
    /// </summary>
    /// <value>
    /// <see cref="FinancialNLPTaskType.Classification"/> (the default) for one prediction per input,
    /// <see cref="FinancialNLPTaskType.TokenClassification"/> or <see cref="FinancialNLPTaskType.NamedEntityRecognition"/>
    /// for one prediction per token.
    /// </value>
    /// <remarks>
    /// <para>
    /// Classification pools the encoder's hidden states and the output is <c>[batch, classes]</c>. Token classification
    /// skips the pooling and applies the head at every position, so the output is <c>[batch, sequence, classes]</c>.
    /// The other task types need a decoder or a span head these models do not have, and are refused by
    /// <see cref="ValidateNLPOptions"/>.
    /// </para>
    /// <para><b>For Beginners:</b> Use Classification to label a whole sentence (for example its sentiment), and
    /// TokenClassification to label each word (for example which words name a company).
    /// </para>
    /// </remarks>
    public FinancialNLPTaskType TaskType { get; set; } = FinancialNLPTaskType.Classification;

    /// <summary>
    /// Gets or sets a factory for the head that sits on the encoder, or null for the default head.
    /// </summary>
    /// <value>A function returning the head's layers, called once each time a model is built; null by default.</value>
    /// <remarks>
    /// <para>
    /// The factory is called for every model built from these options, including clones, so each model owns its own
    /// head. It must return new layer objects on every call. The head's first layer receives the encoder's output:
    /// <c>[batch, hidden]</c> for classification, <c>[batch, sequence, hidden]</c> for token classification.
    /// </para>
    /// <para><b>For Beginners:</b> The default head is dropout followed by one linear layer producing the class
    /// scores. Supply your own to, for example, add a hidden layer or predict several labels at once.
    /// </para>
    /// </remarks>
    public Func<IEnumerable<ILayer<T>>>? ClassificationHeadFactory { get; set; }

    /// <summary>
    /// Gets or sets whether a model built on custom architecture layers adds the default head on top of them.
    /// </summary>
    /// <value>True (the default) to add the default head; false when the architecture's layers already end in one.</value>
    /// <remarks>
    /// <para>
    /// Only consulted when the architecture supplies layers and <see cref="ClassificationHeadFactory"/> is null. With
    /// false the architecture's layers are the whole model, so <c>GetSequenceEmbedding</c> returns their final output.
    /// </para>
    /// <para><b>For Beginners:</b> Set this to false if the layers you pass in already produce class scores.
    /// </para>
    /// </remarks>
    public bool UseDefaultClassificationHead { get; set; } = true;

    /// <summary>
    /// Gets or sets the tokenizer that turns text into token ids, or null for the model's default.
    /// </summary>
    /// <value>A tokenizer whose vocabulary fits the model's vocabulary size; null by default.</value>
    /// <remarks>
    /// <para>
    /// When null, a model loaded from a pretrained checkpoint uses the tokenizer stored with the checkpoint, and any
    /// other model uses its family's default: WordPiece with the bert-base-uncased vocabulary for the BERT models, and a
    /// SentencePiece-style subword tokenizer sized to the vocabulary for the decoder models. Every id the tokenizer can
    /// produce must be below the model's vocabulary size.
    /// </para>
    /// <para><b>For Beginners:</b> The tokenizer must match the vocabulary the model was trained with. Leave this null
    /// unless you trained the model with a different vocabulary.
    /// </para>
    /// </remarks>
    public ITokenizer? Tokenizer { get; set; }

    /// <summary>
    /// Gets or sets whether fine-tuning leaves the whole encoder unchanged and trains only the head.
    /// </summary>
    /// <value>True to train only the head; false (the default) to train every layer.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Freezing the encoder keeps what a pretrained model learned and is much faster, but
    /// usually scores a little lower than fine-tuning everything.
    /// </para>
    /// </remarks>
    public bool FreezeBaseModel { get; set; } = false;

    /// <summary>
    /// Gets or sets how many of the encoder's top transformer blocks fine-tuning trains.
    /// </summary>
    /// <value>-1 (the default) to train every block; N to train only the top N blocks.</value>
    /// <remarks>
    /// <para>
    /// The blocks below the top N, and every encoder layer before them (the embeddings), keep their values; layers after
    /// the last block (final normalization, pooler) and the head still train. Cannot be combined with
    /// <see cref="FreezeBaseModel"/>.
    /// </para>
    /// <para><b>For Beginners:</b> Setting 2 trains only the last two transformer layers plus everything above them.
    /// </para>
    /// </remarks>
    public int NumFineTuneLayers { get; set; } = -1;

    /// <summary>
    /// Checks the shared settings.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when a setting is out of range or the settings contradict each
    /// other.</exception>
    /// <exception cref="NotSupportedException">Thrown when <see cref="TaskType"/> needs a decoder or span head.</exception>
    protected void ValidateNLPOptions()
    {
        if (!Enum.IsDefined(typeof(FinancialNLPTaskType), TaskType))
            throw new ArgumentException("TaskType must be a valid FinancialNLPTaskType value.", nameof(TaskType));
        if (TaskType != FinancialNLPTaskType.Classification && TaskType != FinancialNLPTaskType.TokenClassification
            && TaskType != FinancialNLPTaskType.NamedEntityRecognition)
            throw new NotSupportedException(
                $"TaskType {TaskType} needs a decoder or span head; these models support Classification, "
                + "TokenClassification and NamedEntityRecognition.");
        if (NumFineTuneLayers < -1)
            throw new ArgumentException("NumFineTuneLayers must be -1 (train every block) or at least 0.",
                nameof(NumFineTuneLayers));
        if (FreezeBaseModel && NumFineTuneLayers >= 0)
            throw new ArgumentException(
                "FreezeBaseModel freezes every encoder block, so it cannot be combined with NumFineTuneLayers.",
                nameof(NumFineTuneLayers));
    }
}
