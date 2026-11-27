using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Finance.Enums;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;
using AiDotNet.LossFunctions;
using System.IO;

namespace AiDotNet.Finance.Base;

/// <summary>
/// Base class for all financial NLP models, implementing the dual-mode pattern.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[TensorLayout(TensorAxis.Time, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Classes, Direction = TensorLayoutDirection.Output,
    Note = "Classification: one prediction per input.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Classes, Direction = TensorLayoutDirection.Output,
    Note = "Classification: one prediction per input.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Classes,
    Direction = TensorLayoutDirection.Output, Note = "Token classification: one prediction per token.")]
public abstract partial class FinancialNLPModelBase<T> : FinancialModelBase<T>, IFinancialNLPModel<T>, IShapeContract
{
    /// <inheritdoc/>
    /// <remarks>An architecture may state the maximum sequence length or the hidden width (#2147).</remarks>
    protected override IReadOnlyCollection<int>? ArchitectureInputSizesFromOptions =>
        new[] { MaxSequenceLength, HiddenDimension };

    /// <inheritdoc />
    /// <remarks>
    /// A classification model answers once per input, a token-classification model once per token. Custom architecture
    /// layers or a custom head decide the output themselves, so no contract is published for them.
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        bool custom = (Architecture.Layers is not null && Architecture.Layers.Count > 0)
            || LanguageModelOptions.ClassificationHeadFactory is not null;
        if (custom || (inputRank != 1 && inputRank != 2))
            return null;
        var classes = new OutputAxisContract(TensorAxis.Classes, AxisRelation.Fixed(_baseNumSentimentClasses));
        var axes = new List<OutputAxisContract>();
        if (inputRank == 2)
            axes.Add(new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)));
        if (!IsSequenceLevelTask)
            axes.Add(new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)));
        axes.Add(classes);
        return axes;
    }

    /// <summary>
    /// Declares the token-id geometry this model accepts, in place of the [batch, sequence,
    /// features] geometry a numeric financial model publishes.
    /// </summary>
    /// <remarks>
    /// The public surface of an NLP model takes token ids - <c>Tokenize</c> emits them and
    /// <c>AnalyzeSentiment</c> consumes them - which is why the axis contract above gives a
    /// meaning to rank 1 and rank 2 and to nothing else, and why the layout attributes on this
    /// class declare Time and Batch-by-Time inputs. Inheriting the numeric constraint instead
    /// published a third, feature-width axis that no caller on this surface can supply, so a
    /// generic caller conforming to the published constraint built an input the model refused.
    /// </remarks>
    public override ModelInputShapeConstraint GetInputShapeConstraint()
        => CrossSectionalInputConstraint;

    protected readonly int _baseMaxSequenceLength;
    protected readonly int _baseVocabularySize;
    protected readonly int _baseHiddenDimension;
    protected readonly int _baseNumSentimentClasses;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The maximum number of tokens the model can read at once.
    /// Longer texts are truncated or split to fit this length.
    /// </para>
    /// </remarks>
    public virtual int MaxSequenceLength => _baseMaxSequenceLength;

    /// <inheritdoc/>
    /// <remarks>
    /// The model reads at most <see cref="MaxSequenceLength"/> tokens: a longer [Time] or [Batch, Time] input keeps its
    /// first <see cref="MaxSequenceLength"/> positions, as a tokenizer truncating to the context window would. Training
    /// forwards through the same forecast, so both see the same window.
    /// </remarks>
    protected override Tensor<T> ForecastCore(Tensor<T> input, double[]? quantiles)
        => base.ForecastCore(TruncateToContextWindow(input), quantiles);

    /// <summary>
    /// The first <see cref="MaxSequenceLength"/> positions of a [Time] or [Batch, Time] token input; other inputs, and
    /// inputs that already fit, are returned unchanged.
    /// </summary>
    private Tensor<T> TruncateToContextWindow(Tensor<T> input)
    {
        int limit = MaxSequenceLength;
        if (limit <= 0 || (input.Rank != 1 && input.Rank != 2)) return input;
        int timeAxis = input.Rank - 1;
        return input.Shape[timeAxis] <= limit ? input : Engine.TensorNarrow(input, timeAxis, 0, limit);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The size of the vocabulary (how many unique tokens the model knows).
    /// Larger vocabularies allow more precise language understanding but require more memory.
    /// </para>
    /// </remarks>
    public virtual int VocabularySize => _baseVocabularySize;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The embedding size used to represent each token.
    /// Think of this as the number of features describing each word.
    /// </para>
    /// </remarks>
    public virtual int HiddenDimension => _baseHiddenDimension;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many sentiment categories the model can output
    /// (e.g., negative/neutral/positive).
    /// </para>
    /// </remarks>
    public virtual int NumSentimentClasses => _baseNumSentimentClasses;

    /// <summary>
    /// Initializes a new NLP model base for training (native mode).
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="maxSequenceLength">Maximum token sequence length.</param>
    /// <param name="vocabularySize">Vocabulary size.</param>
    /// <param name="hiddenDimension">Embedding dimension.</param>
    /// <param name="numSentimentClasses">Number of sentiment classes.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up a trainable NLP model.
    /// It defines how long text inputs can be, how big the vocabulary is,
    /// and how large the internal embeddings are.
    /// </para>
    /// </remarks>
    protected FinancialNLPModelBase(
        NeuralNetworkArchitecture<T> architecture,
        int maxSequenceLength,
        int vocabularySize,
        int hiddenDimension,
        int numSentimentClasses = 3,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, maxSequenceLength, 1, architecture.InputSize, lossFunction)
    {
        if (maxSequenceLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxSequenceLength), "Must be positive.");
        if (vocabularySize <= 0)
            throw new ArgumentOutOfRangeException(nameof(vocabularySize), "Must be positive.");
        if (hiddenDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(hiddenDimension), "Must be positive.");
        if (numSentimentClasses <= 0)
            throw new ArgumentOutOfRangeException(nameof(numSentimentClasses), "Must be positive.");

        _baseMaxSequenceLength = maxSequenceLength;
        _baseVocabularySize = vocabularySize;
        _baseHiddenDimension = hiddenDimension;
        _baseNumSentimentClasses = numSentimentClasses;
        Options = new FinancialNLPOptions();
    }

    /// <summary>
    /// Initializes a new NLP model base from a pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="maxSequenceLength">Maximum token sequence length.</param>
    /// <param name="vocabularySize">Vocabulary size.</param>
    /// <param name="hiddenDimension">Embedding dimension.</param>
    /// <param name="numSentimentClasses">Number of sentiment classes.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you have a pretrained language model
    /// and want fast inference without training.
    /// </para>
    /// </remarks>
    protected FinancialNLPModelBase(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int maxSequenceLength,
        int vocabularySize,
        int hiddenDimension,
        int numSentimentClasses = 3)
        : base(architecture, onnxModelPath, maxSequenceLength, 1, architecture.InputSize)
    {
        if (maxSequenceLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxSequenceLength), "Must be positive.");
        if (vocabularySize <= 0)
            throw new ArgumentOutOfRangeException(nameof(vocabularySize), "Must be positive.");
        if (hiddenDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(hiddenDimension), "Must be positive.");
        if (numSentimentClasses <= 0)
            throw new ArgumentOutOfRangeException(nameof(numSentimentClasses), "Must be positive.");

        _baseMaxSequenceLength = maxSequenceLength;
        _baseVocabularySize = vocabularySize;
        _baseHiddenDimension = hiddenDimension;
        _baseNumSentimentClasses = numSentimentClasses;
        Options = new FinancialNLPOptions();
    }

    #region Encoder and head

    /// <summary>
    /// The number of classes a model predicts: the architecture's stated output size, or the options' count when the
    /// architecture states none.
    /// </summary>
    /// <param name="architecture">The model's architecture.</param>
    /// <param name="optionsClassCount">The class count the options declare.</param>
    /// <returns>The class count.</returns>
    protected static int ResolveClassCount(NeuralNetworkArchitecture<T> architecture, int optionsClassCount)
    {
        if (architecture is null) throw new ArgumentNullException(nameof(architecture));
        return architecture.OutputSize > 0 ? architecture.OutputSize : optionsClassCount;
    }


    /// <summary>
    /// The number of leading <see cref="NeuralNetworkBase{T}.Layers"/> that form the encoder; the layers after them
    /// are the head. -1 until <see cref="InitializeEncoderAndHead"/> has run, meaning the whole stack is the encoder.
    /// </summary>
    private int _encoderLayerCount = -1;

    /// <summary>Class names a loaded checkpoint declared (its id2label), or null for the default names.</summary>
    private string[]? _classLabels;

    /// <summary>The tokenizer <see cref="Tokenize"/> uses, resolved on first use.</summary>
    [Scratch]
    private ITokenizer? _resolvedTokenizer;

    /// <summary>
    /// The shared financial-language-model settings: task, head, tokenizer and fine-tuning freezes.
    /// </summary>
    protected abstract FinancialLanguageModelOptions<T> LanguageModelOptions { get; }

    /// <summary>
    /// The position whose hidden state summarizes a sequence: the first ([CLS]) for a bidirectional encoder, the last for
    /// a causal decoder, whose last position is the only one that has read every token.
    /// </summary>
    protected abstract SequenceTokenSliceLayer<T>.Position SequencePoolingPosition { get; }

    /// <summary>
    /// Builds the tokenizer used when <see cref="FinancialLanguageModelOptions{T}.Tokenizer"/> is null.
    /// </summary>
    /// <returns>A tokenizer whose ids fit <see cref="VocabularySize"/>.</returns>
    protected abstract ITokenizer CreateDefaultTokenizer();

    /// <summary>
    /// The number of leading layers that form the encoder; the rest are the head.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The encoder ends at the sequence embedding (the pooled vector for classification, the per-token hidden states for
    /// token classification); <see cref="GetSequenceEmbedding"/> runs exactly these layers. Every layer, encoder and
    /// head alike, sits in the one <see cref="NeuralNetworkBase{T}.Layers"/> list, so saving, cloning and the parameter
    /// vector cover the whole model.
    /// </para>
    /// </remarks>
    protected int EncoderLayerCount =>
        _encoderLayerCount < 0 || _encoderLayerCount > Layers.Count ? Layers.Count : _encoderLayerCount;

    /// <summary>
    /// Whether the model predicts once per input (pooling the encoder) rather than once per token.
    /// </summary>
    protected bool IsSequenceLevelTask => LanguageModelOptions.TaskType == FinancialNLPTaskType.Classification;

    /// <summary>
    /// Fills <see cref="NeuralNetworkBase{T}.Layers"/> with the encoder followed by the head.
    /// </summary>
    /// <param name="defaultEncoder">Builds the model's own encoder; used when the architecture supplies no layers.</param>
    /// <param name="defaultHead">Builds the model's own head; used unless
    /// <see cref="FinancialLanguageModelOptions{T}.ClassificationHeadFactory"/> supplies one, or the architecture's
    /// layers are the whole model.</param>
    /// <remarks>
    /// <para>
    /// Layers passed through the architecture replace the encoder. The head is then the factory's layers, the default
    /// head, or nothing when <see cref="FinancialLanguageModelOptions{T}.UseDefaultClassificationHead"/> is false.
    /// </para>
    /// </remarks>
    protected void InitializeEncoderAndHead(
        Func<IEnumerable<ILayer<T>>> defaultEncoder,
        Func<IEnumerable<ILayer<T>>> defaultHead)
    {
        if (defaultEncoder is null) throw new ArgumentNullException(nameof(defaultEncoder));
        if (defaultHead is null) throw new ArgumentNullException(nameof(defaultHead));

        var options = LanguageModelOptions;
        var customLayers = Architecture.Layers;
        bool custom = customLayers is not null && customLayers.Count > 0;
        if (customLayers is not null && custom)
            Layers.AddRange(customLayers);
        else
            Layers.AddRange(defaultEncoder());
        _encoderLayerCount = Layers.Count;

        IEnumerable<ILayer<T>>? head;
        if (options.ClassificationHeadFactory is { } factory)
        {
            head = factory()
                ?? throw new InvalidOperationException("ClassificationHeadFactory returned null; return the head's layers.");
        }
        else
        {
            head = custom && !options.UseDefaultClassificationHead ? null : defaultHead();
        }

        if (head is not null)
        {
            foreach (var layer in head)
            {
                if (layer is null)
                    throw new InvalidOperationException("The classification head contains a null layer.");
                Layers.Add(layer);
            }
        }

        if (custom)
            ValidateCustomLayersWithOwnershipRollback(Layers);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// In inference this runs the eager executor, which streams the weights of a model too large to hold in memory at
    /// once. The training forward runs the same layers in order, in training mode, so dropout stays on.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastNative(Tensor<T> input, double[]? quantiles)
    {
        if (!IsTrainingMode)
        {
            TryAutoEnableWeightStreaming(isTrainingOverride: false);
            return PredictEager(input);
        }

        return RunLayerSpan(input, 0, Layers.Count);
    }

    /// <summary>
    /// Runs the encoder on token ids, in inference mode, and returns its output.
    /// </summary>
    private Tensor<T> RunEncoder(Tensor<T> tokenIds, out int sequenceAxis)
    {
        if (tokenIds is null) throw new ArgumentNullException(nameof(tokenIds));
        if (!UseNativeMode)
            throw new InvalidOperationException("Embeddings need the native layers; an ONNX model only exposes its output.");
        var input = TruncateToContextWindow(tokenIds);
        ValidateInputShape(input);
        sequenceAxis = input.Rank - 1;
        return InInferenceMode(() => RunLayerSpan(input, 0, EncoderLayerCount));
    }

    #endregion

    #region IFinancialNLPModel

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Returns softmax probabilities over the classes: the model's logits, normalized along the class axis.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the simplest sentiment interface:
    /// provide token IDs and the model returns how likely each class is.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> AnalyzeSentiment(Tensor<T> tokenIds)
    {
        var logits = Predict(tokenIds);
        return Engine.Softmax(logits, axis: logits.Rank - 1);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Each text is tokenized with <see cref="Tokenize"/> and classified on its own, so no padding enters the model.
    /// <see cref="SentimentResult{T}.ClassProbabilities"/> holds softmax probabilities and
    /// <see cref="SentimentResult{T}.Confidence"/> the probability of the predicted class. Class names come from a
    /// loaded checkpoint's labels when it declares them.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This helper accepts raw text strings, tokenizes them,
    /// runs the model, and returns a readable sentiment result for each input.
    /// </para>
    /// </remarks>
    public virtual SentimentResult<T>[] AnalyzeSentiment(string[] texts)
    {
        if (texts is null)
            throw new ArgumentNullException(nameof(texts));

        var results = new SentimentResult<T>[texts.Length];
        for (int i = 0; i < texts.Length; i++)
        {
            var tokenIds = Tokenize(texts[i]);
            if (tokenIds.Length == 0)
            {
                results[i] = new SentimentResult<T>
                {
                    OriginalText = texts[i],
                    PredictedClass = "unknown",
                    Confidence = NumOps.Zero,
                    ClassProbabilities = new Dictionary<string, T>()
                };
                continue;
            }

            // [1, sequence_length]: the models take a batch axis.
            var inputTensor = new Tensor<int>(new[] { 1, tokenIds.Length }, new Vector<int>(tokenIds)).Cast<T>();
            var probabilities = AnalyzeSentiment(inputTensor);
            int classCount = probabilities.Shape[probabilities.Rank - 1];
            if (probabilities.Length == 0 || probabilities.Length != classCount)
                throw new InvalidOperationException(
                    $"AnalyzeSentiment(string[]) needs one prediction per text, but the model returned shape "
                    + $"[{string.Join(", ", probabilities.Shape.ToArray())}]. A token-classification model labels each "
                    + "token instead; call Predict on the token ids.");

            var scores = Engine.Reshape(probabilities, new[] { classCount });
            int bestIdx = Engine.TensorArgMax(scores, 0).Data.Span[0];
            var byLabel = new Dictionary<string, T>(classCount);
            var span = scores.Data.Span;
            for (int c = 0; c < classCount; c++)
                byLabel[GetClassLabel(c, classCount)] = span[c];

            results[i] = new SentimentResult<T>
            {
                OriginalText = texts[i],
                PredictedClass = GetClassLabel(bestIdx, classCount),
                Confidence = span[bestIdx],
                ClassProbabilities = byLabel
            };
        }

        return results;
    }

    /// <summary>
    /// Records the class names a loaded checkpoint declares, in class-index order.
    /// </summary>
    /// <param name="labels">One name per class.</param>
    protected void SetClassLabels(IReadOnlyList<string> labels)
    {
        if (labels is null) throw new ArgumentNullException(nameof(labels));
        _classLabels = labels.ToArray();
    }

    private string GetClassLabel(int index, int classCount)
    {
        if (_classLabels is not null && _classLabels.Length == classCount)
            return _classLabels[index];
        if (classCount == 3)
        {
            return index switch
            {
                0 => "negative",
                1 => "neutral",
                2 => "positive",
                _ => $"class_{index}"
            };
        }

        return $"class_{index}";
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// The contextual hidden state of every token: the last encoder output that still has a sequence axis,
    /// <c>[batch, sequence, hidden]</c> for a <c>[batch, sequence]</c> input. For the default layers that is the output
    /// of the last transformer block (after the final normalization of a pre-norm stack), before any pooling.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Embeddings are dense numeric representations of text
    /// that capture meaning. This returns one vector per token, each informed by the tokens around it.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> GetEmbeddings(Tensor<T> tokenIds)
    {
        if (tokenIds is null) throw new ArgumentNullException(nameof(tokenIds));
        if (!UseNativeMode)
            throw new InvalidOperationException("Embeddings need the native layers; an ONNX model only exposes its output.");
        var input = TruncateToContextWindow(tokenIds);
        ValidateInputShape(input);
        int tokenRank = input.Rank + 1;
        return InInferenceMode(() =>
        {
            Tensor<T>? lastTokenLevel = null;
            var current = input;
            for (int i = 0; i < EncoderLayerCount; i++)
            {
                current = Layers[i].Forward(current);
                if (current.Rank == tokenRank)
                    lastTokenLevel = current;
            }

            return lastTokenLevel
                ?? throw new InvalidOperationException(
                    "No encoder layer produced per-token hidden states [batch, sequence, hidden].");
        });
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// The encoder's output: for classification the pooled vector the head reads, <c>[batch, hidden]</c> (BERT's pooler
    /// output, or a decoder's last-position hidden state). When the encoder ends at per-token states (token
    /// classification, or custom layers without pooling) the state at <see cref="SequencePoolingPosition"/> is taken.
    /// When the architecture's layers are the whole model, this is their final output.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This summarizes an entire sequence into a single embedding,
    /// which can be used for classification or similarity search.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> GetSequenceEmbedding(Tensor<T> tokenIds)
    {
        var encoded = RunEncoder(tokenIds, out int sequenceAxis);
        if (encoded.Rank != sequenceAxis + 2)
            return encoded;
        int length = encoded.Shape[sequenceAxis];
        int position = SequencePoolingPosition == SequenceTokenSliceLayer<T>.Position.First ? 0 : length - 1;
        return Engine.TensorSliceAxis(encoded, axis: sequenceAxis, index: position);
    }

    /// <summary>
    /// The tokenizer <see cref="Tokenize"/> and <see cref="Detokenize"/> use.
    /// </summary>
    /// <remarks>
    /// <see cref="FinancialLanguageModelOptions{T}.Tokenizer"/> when set, otherwise the model's default, built on first use.
    /// </remarks>
    protected ITokenizer Tokenizer => _resolvedTokenizer ??= LanguageModelOptions.Tokenizer ?? CreateDefaultTokenizer();

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Encodes the text with <see cref="Tokenizer"/>, adding the model's special tokens and truncating to
    /// <paramref name="maxLength"/> (or <see cref="MaxSequenceLength"/>). The ids are not padded: one text is one
    /// sequence of its own length, so no padding token is attended to.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Tokenization turns text into integers the model can read. Words the vocabulary does not
    /// have are split into smaller known pieces.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when the tokenizer produces an id outside the model's
    /// vocabulary.</exception>
    public virtual int[] Tokenize(string text, int? maxLength = null)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        int limit = maxLength ?? MaxSequenceLength;
        if (limit < 1) throw new ArgumentOutOfRangeException(nameof(maxLength), "Must be at least 1.");

        var encoded = Tokenizer.Encode(text, new EncodingOptions
        {
            AddSpecialTokens = true,
            MaxLength = limit,
            Truncation = true,
            Padding = false,
        });
        var ids = encoded.TokenIds.ToArray();
        for (int i = 0; i < ids.Length; i++)
        {
            if (ids[i] < 0 || ids[i] >= VocabularySize)
                throw new InvalidOperationException(
                    $"The tokenizer produced id {ids[i]}, outside this model's vocabulary of {VocabularySize}. Set "
                    + "Tokenizer to one whose vocabulary matches the model, or raise VocabularySize.");
        }

        return ids;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Detokenization converts token IDs back into text, dropping special tokens such as [CLS].
    /// </para>
    /// </remarks>
    public virtual string Detokenize(int[] tokenIds)
    {
        if (tokenIds is null) throw new ArgumentNullException(nameof(tokenIds));
        return Tokenizer.Decode(tokenIds.ToList(), skipSpecialTokens: true);
    }

    #endregion

    #region Fine-tuning freezes

    /// <inheritdoc/>
    /// <remarks>
    /// <see cref="FinancialLanguageModelOptions{T}.FreezeBaseModel"/> freezes every encoder layer;
    /// <see cref="FinancialLanguageModelOptions{T}.NumFineTuneLayers"/> freezes the encoder layers up to and including
    /// the last transformer block outside the top N. Frozen layers keep their values through training but stay in the
    /// saved, cloned and counted parameters.
    /// </remarks>
    protected override AiDotNet.Models.Parameters.ParameterSlotRole ResolveLayerParameterRole(
        int layerIndex, AiDotNet.Models.Parameters.ParameterSlotRole declared)
    {
        if (declared != AiDotNet.Models.Parameters.ParameterSlotRole.Trainable)
            return declared;
        return layerIndex < FrozenEncoderLayerCount()
            ? AiDotNet.Models.Parameters.ParameterSlotRole.Frozen
            : declared;
    }

    /// <inheritdoc/>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        int frozenCount = FrozenEncoderLayerCount();
        if (frozenCount == 0) return parameters;
        var frozen = new HashSet<Tensor<T>>(
            AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(Layers.Take(frozenCount).ToList(), -1),
            AiDotNet.Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(parameter => !frozen.Contains(parameter)).ToList();
    }

    /// <summary>
    /// How many leading layers fine-tuning leaves unchanged.
    /// </summary>
    private int FrozenEncoderLayerCount()
    {
        var options = LanguageModelOptions;
        int encoder = EncoderLayerCount;
        if (options.FreezeBaseModel) return encoder;
        if (options.NumFineTuneLayers < 0) return 0;

        var blocks = new List<int>();
        for (int i = 0; i < encoder; i++)
        {
            if (IsTransformerBlock(Layers[i])) blocks.Add(i);
        }

        int frozenBlocks = blocks.Count - options.NumFineTuneLayers;
        if (frozenBlocks <= 0) return 0;
        // Everything up to and including the last frozen block: the embeddings and the lower blocks.
        return blocks[frozenBlocks - 1] + 1;
    }

    private static bool IsTransformerBlock(ILayer<T> layer)
        => layer is TransformerEncoderBlock<T> or PreLNTransformerBlock<T> or TransformerDecoderBlock<T>;

    #endregion

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Adds NLP-specific configuration to the standard finance metrics,
    /// such as vocabulary size and maximum sequence length.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        var metrics = base.GetFinancialMetrics();
        metrics["MaxSequenceLength"] = NumOps.FromDouble(MaxSequenceLength);
        metrics["VocabularySize"] = NumOps.FromDouble(VocabularySize);
        metrics["HiddenDimension"] = NumOps.FromDouble(HiddenDimension);
        metrics["NumSentimentClasses"] = NumOps.FromDouble(NumSentimentClasses);

        return metrics;
    }
}
