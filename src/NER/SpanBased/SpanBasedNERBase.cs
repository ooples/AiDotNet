using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NER.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Optimizers;

namespace AiDotNet.NER.SpanBased;

/// <summary>
/// Base class for span-based NER models (SpERT, BiaffineNER, PURE).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Span-based NER models enumerate candidate entity spans (contiguous subsequences) and
/// classify each span as an entity type or non-entity. This approach differs fundamentally
/// from sequence labeling (BIO tagging):
///
/// <b>Sequence Labeling (BiLSTM-CRF):</b>
/// Labels each token independently: [B-PER, I-PER, O, O, B-ORG, I-ORG]
/// Cannot naturally handle nested entities (e.g., "New York" inside "New York University")
///
/// <b>Span-Based (SpERT, BiaffineNER, PURE):</b>
/// Enumerates spans: (0,1)="Barack", (0,2)="Barack Obama", (2,3)="was", ...
/// Classifies each span: (0,2)=PER, (4,6)=LOC, (0,6)=non-entity, ...
/// Naturally handles nested entities because different spans can have different labels
///
/// <b>Architecture:</b>
/// <code>
///   [token embeddings] --> [Encoder (Transformer/BiLSTM)] --> [Span Representation] --> [Span Classifier] --> [Entity Spans]
/// </code>
///
/// The span representation combines boundary tokens, span content, and span width features
/// into a fixed-size vector for classification.
/// </para>
/// </remarks>
public abstract partial class SpanBasedNERBase<T> : SequenceLabeling.SequenceLabelingNERBase<T>, INERModel<T>
{
    #region Fields

    private readonly SpanBasedNEROptions _options;
    private bool _useNativeMode;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _disposed;
    private readonly string _modelName;
    private readonly string _paperCitation;

    #endregion

    #region INERModel Properties

    /// <inheritdoc />
    public int[] ExpectedInputShape => [_options.MaxSequenceLength, _options.HiddenDimension];

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a span-based NER model in ONNX inference mode.
    /// </summary>
    protected SpanBasedNERBase(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        SpanBasedNEROptions options,
        string modelName,
        string paperCitation)
        : base(architecture)
    {
        _options = options;
        _options.ModelPath = modelPath;
        _useNativeMode = false;
        _modelName = modelName;
        _paperCitation = paperCitation;
        ValidateOptions();
        ApplyOptionsToBase();
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
    }

    /// <summary>
    /// Creates a span-based NER model in native training mode.
    /// </summary>
    protected SpanBasedNERBase(
        NeuralNetworkArchitecture<T> architecture,
        SpanBasedNEROptions options,
        string modelName,
        string paperCitation,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options;
        _useNativeMode = true;
        _modelName = modelName;
        _paperCitation = paperCitation;
        ValidateOptions();
        ApplyOptionsToBase();

        // Yu et al. 2020 (arXiv:2005.07150) Table 1 specifies Adam at 1e-3 for the biaffine span
        // scorer, and the sibling span architectures follow it. Constructing AdamW here applied a
        // decoupled weight decay of 0.01 -- AdamW's own default, since only the learning rate was
        // supplied -- to every parameter on every step, which no span-NER paper asks for. With the
        // biaffine tensor's large fan-in that decay drove the parameters to NaN in a single step
        // (measured: L2 42.2154 -> NaN), which is what OptimizerStep_ParamL2_DoesNotExplode reports
        // and its message predicted: "weight decay too aggressive".
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                // The reference decays the rate as training proceeds
                // (juntaoy/biaffine-ner, experiments.conf: decay_rate = 0.999 applied every
                // decay_frequency = 100 steps). The paper describes no schedule at all, so without
                // this the rate stayed at its initial value for the whole run.
                LearningRateScheduler = new StepLRScheduler(
                    _options.LearningRate, _options.DecayFrequency, _options.DecayRate),
                SchedulerStepMode = SchedulerStepMode.StepPerBatch
            });

        InitializeLayers();
    }

    #endregion

    #region Properties

    /// <summary>
    /// Gets the options for this span-based NER model.
    /// </summary>
    protected SpanBasedNEROptions NEROptions => _options;

    /// <summary>
    /// Gets whether this model is in native training mode.
    /// </summary>
    protected bool UseNativeMode => _useNativeMode;

    #endregion

    #region Sequence Labeling

    /// <inheritdoc />
    public override Tensor<T> PredictLabels(Tensor<T> tokenEmbeddings)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessTokens(tokenEmbeddings);
        var output = IsOnnxMode ? RunOnnxInference(preprocessed) : Forward(preprocessed);
        return PostprocessOutput(output);
    }

    /// <inheritdoc />
    protected override Tensor<T> ComputeEmissionScores(Tensor<T> tokenEmbeddings)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
        {
            return RunOnnxInference(tokenEmbeddings);
        }

        Tensor<T> output = tokenEmbeddings;
        foreach (var layer in Layers)
        {
            if (layer is ConditionalRandomFieldLayer<T>)
                break;
            output = layer.Forward(output);
        }

        return output;
    }

    #endregion

    #region INERModel Methods

    /// <inheritdoc />
    Task INERModel<T>.TrainAsync(
        Tensor<T> tokenEmbeddings,
        Tensor<T> labels,
        int epochs,
        IProgress<NERTrainingProgress>? progress,
        CancellationToken cancellationToken)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (_optimizer is null) throw new InvalidOperationException("Optimizer is not initialized.");

        return Task.Run(() =>
        {
            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                SetTrainingMode(true);
                try
                {
                    var preprocessed = PreprocessTokens(tokenEmbeddings);
                    var preprocessedLabels = PreprocessLabels(labels, preprocessed.Shape[0]);
                    var output = Forward(preprocessed);
                    double loss = NumOps.ToDouble(LossFunction.CalculateLoss(
                        output.ToVector(), preprocessedLabels.ToVector()));
                    // Backward removed — tape-based training handles gradients
                    _optimizer.UpdateParameters(Layers);

                    progress?.Report(new NERTrainingProgress
                    {
                        CurrentEpoch = epoch,
                        TotalEpochs = epochs,
                        CurrentBatch = 1,
                        TotalBatches = 1,
                        Loss = loss
                    });
                }
                finally
                {
                    SetTrainingMode(false);
                }
            }
        }, cancellationToken);
    }

    /// <inheritdoc />
    IEnumerable<Tensor<T>> INERModel<T>.PredictBatch(IEnumerable<Tensor<T>> sequences)
    {
        foreach (var seq in sequences)
        {
            yield return PredictLabels(seq);
        }
    }

    /// <inheritdoc />
    void INERModel<T>.ValidateInputShape(Tensor<T> input)
    {
        if (input.Rank < 2 || input.Rank > 3)
            throw new ArgumentException(
                $"Expected rank-2 [seqLen, hiddenDim] or rank-3 [batch, seqLen, hiddenDim] tensor. Got rank {input.Rank}.");

        int embDim = input.Rank == 2 ? input.Shape[1] : input.Shape[2];
        if (embDim != _options.HiddenDimension)
            throw new ArgumentException(
                $"Hidden dimension mismatch. Expected {_options.HiddenDimension}, got {embDim}.");

        int seqDim = input.Rank == 2 ? input.Shape[0] : input.Shape[1];
        if (seqDim > _options.MaxSequenceLength)
            throw new ArgumentException(
                $"Sequence length {seqDim} exceeds maximum {_options.MaxSequenceLength}.");
    }

    /// <inheritdoc />
    string INERModel<T>.GetModelSummary()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"=== {_modelName} Model Summary ===");
        sb.AppendLine($"Type: Span-Based NER");
        sb.AppendLine($"Variant: {_options.Variant}");
        sb.AppendLine($"Mode: {(IsOnnxMode ? "ONNX Inference" : "Native Training")}");
        sb.AppendLine($"Hidden Dimension: {_options.HiddenDimension}");
        sb.AppendLine($"Attention Heads: {_options.NumAttentionHeads}");
        sb.AppendLine($"Transformer Layers: {_options.NumTransformerLayers}");
        sb.AppendLine($"Max Span Length: {_options.MaxSpanLength}");
        sb.AppendLine($"Span Embedding Dim: {_options.SpanEmbeddingDimension}");
        sb.AppendLine($"Num Labels: {_options.NumLabels}");
        sb.AppendLine($"Max Sequence Length: {_options.MaxSequenceLength}");
        sb.AppendLine($"Dropout Rate: {_options.DropoutRate}");
        sb.AppendLine($"Learning Rate: {_options.LearningRate}");
        sb.AppendLine($"Labels: {string.Join(", ", _options.LabelNames)}");
        sb.AppendLine($"Total Layers: {Layers.Count}");

        for (int i = 0; i < Layers.Count; i++)
        {
            sb.AppendLine($"  Layer {i}: {Layers[i].GetType().Name}");
        }

        return sb.ToString();
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(CreateDefaultLayers());
        }
    }

    /// <summary>
    /// Creates the default layer stack for this span-based NER model.
    /// </summary>
    protected abstract IEnumerable<ILayer<T>> CreateDefaultLayers();

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (_optimizer is null) throw new InvalidOperationException("Optimizer is not initialized.");
        SetTrainingMode(true);
        try
        {
            // Train on the input's NATURAL sequence length and align the labels
            // to it. The transformer encoder is length-agnostic, so there is no
            // need to pad up to MaxSequenceLength here (the way the inference
            // PreprocessTokens does). Two bugs are avoided by NOT padding:
            //   1. Shape contract: training previously fed the RAW [seqLen, hid]
            //      input to TrainWithTape -> [seqLen, numLabels] logits, while the
            //      label tensor was shaped from Predict's PADDED output
            //      [MaxSequenceLength]. CrossEntropyWithLogitsLoss then indexed
            //      past its one-hot buffer (IndexOutOfRange). Aligning the labels
            //      to the logits' natural length fixes this.
            //   2. Loss quality + cost: padding to MaxSequenceLength would make
            //      the forward process MaxSequenceLength positions whose
            //      (zero-input, arbitrary-label) pairs are unlearnable noise that
            //      pollutes the loss, and would multiply the per-step cost by
            //      ~MaxSequenceLength / seqLen. Training on the real tokens keeps
            //      the loss meaningful and the step cheap.

            // Canonicalize and validate input, matching the preprocessing path used
            // by PredictLabels. This ensures unsupported tensor ranks are rejected
            // and sequences are truncated to MaxSequenceLength before label alignment.
            var preprocessed = PreprocessTokens(input);
            int validatedSeqLen = preprocessed.Rank == 3 ? preprocessed.Shape[1] : preprocessed.Shape[0];
            var alignedExpected = BuildTrainingTargets(expected, validatedSeqLen);
            // Use the model's configured optimizer. Falling back to NeuralNetworkBase's
            // default Adam step ignores SpanBasedNEROptions.LearningRate (5e-5 for
            // SpERT) and is large enough to make the generated memorization test diverge.
            TrainWithTape(preprocessed, alignedExpected, _optimizer);
        }
        finally { SetTrainingMode(false); }
    }

    /// <summary>
    /// Trains on span-level supervision: a category per candidate span, which is the annotation
    /// form these architectures are actually defined over.
    /// </summary>
    /// <param name="tokenEmbeddings">Token representations, <c>[seqLen, hidden]</c> or <c>[batch, seqLen, hidden]</c>.</param>
    /// <param name="spanTargets">
    /// Category indices per span, <c>[seqLen * seqLen]</c> or <c>[batch, seqLen * seqLen]</c>,
    /// indexed as <c>start * seqLen + end</c>. Use <c>0</c> for the non-entity class and <c>-1</c>
    /// for a position that is not a candidate at all (<c>end &lt; start</c>, or longer than the
    /// configured maximum span), which is ignored by the loss rather than taught as a negative.
    /// </param>
    /// <remarks>
    /// <para>
    /// <see cref="Train"/> takes per-TOKEN labels, which is the right interface for a
    /// sequence-labelling corpus but cannot express what a span model predicts: a token label
    /// only ever describes a single-token entity, so the derived supervision reaches the diagonal
    /// of the l x l grid and every multi-token span is taught as a non-entity. The standard span
    /// corpora (CoNLL-2003, ACE, GENIA) ship entity spans, and the reference implementations of
    /// these architectures read them directly.
    /// </para>
    /// <para>
    /// The signal is an explicit method rather than something inferred from the target's shape.
    /// Inference is not safe here: a span grid's trailing axis is <c>seqLen * seqLen</c>, which is
    /// exactly the shape of this model's own output, so a caller passing an output-shaped tensor
    /// for any other reason would be silently reinterpreted as span annotations.
    /// </para>
    /// </remarks>
    public void TrainWithSpanTargets(Tensor<T> tokenEmbeddings, Tensor<T> spanTargets)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");

        SetTrainingMode(true);
        try
        {
            var preprocessed = PreprocessTokens(tokenEmbeddings);
            int seqLen = preprocessed.Rank == 3 ? preprocessed.Shape[1] : preprocessed.Shape[0];

            int trailing = spanTargets.Rank >= 1 ? spanTargets.Shape[spanTargets.Rank - 1] : 0;
            if (trailing != seqLen * seqLen)
            {
                throw new ArgumentException(
                    $"Span targets must carry one category per span: expected a trailing axis of " +
                    $"{seqLen * seqLen} ({seqLen} x {seqLen}) for a sequence of {seqLen} tokens, got {trailing}.",
                    nameof(spanTargets));
            }

            TrainWithTape(preprocessed, spanTargets, _optimizer);
        }
        finally { SetTrainingMode(false); }
    }

    /// <summary>
    /// Shapes the supervision the model is trained against, given token-level labels.
    /// </summary>
    /// <remarks>
    /// Defaults to per-token labels aligned to the encoder's sequence length, which is what a
    /// token-classification head consumes. Span-scoring heads that emit one score per candidate
    /// span override this to build span-level supervision instead.
    /// </remarks>
    /// <param name="labels">The caller's token-level labels.</param>
    /// <param name="seqLen">The encoder's validated sequence length.</param>
    /// <returns>The target tensor to pair with the model's output.</returns>
    protected virtual Tensor<T> BuildTrainingTargets(Tensor<T> labels, int seqLen)
        => PreprocessLabels(labels, seqLen);

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    protected override Tensor<T> PreprocessTokens(Tensor<T> rawEmbeddings)
    {
        int maxLen = _options.MaxSequenceLength;
        int hidDim = _options.HiddenDimension;

        // Reject unsupported tensor ranks instead of falling through to rank-2 path
        if (rawEmbeddings.Rank < 2 || rawEmbeddings.Rank > 3)
            throw new ArgumentException(
                $"Expected rank-2 [seqLen, hiddenDim] or rank-3 [batch, seqLen, hiddenDim] tensor. Got rank {rawEmbeddings.Rank}.");

        // Process the input's NATURAL sequence length: the transformer encoder is
        // length-agnostic, so only TRUNCATE sequences that exceed the configured
        // MaxSequenceLength — never pad shorter ones UP to it. Padding a single
        // sequence up to MaxSequenceLength multiplies the per-forward cost by
        // ~MaxSequenceLength / seqLen (e.g. 256/8 = 32x) for zero benefit, and on
        // the training path it pollutes the loss with zero-input positions. The
        // Train() override above already operates on the natural length, so this
        // keeps inference and training on matching shapes.

        // Handle rank-3 [batch, seqLen, hidDim]
        if (rawEmbeddings.Rank == 3)
        {
            int batch = rawEmbeddings.Shape[0];
            int seqLen3 = rawEmbeddings.Shape[1];
            if (seqLen3 <= maxLen) return rawEmbeddings;

            var truncated3 = new Tensor<T>([batch, maxLen, hidDim]);
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < maxLen; s++)
                    for (int d = 0; d < hidDim; d++)
                        truncated3[b, s, d] = rawEmbeddings[b, s, d];
            return truncated3;
        }

        // Rank-2 [seqLen, hidDim]
        int seqLen = rawEmbeddings.Shape[0];
        if (seqLen <= maxLen) return rawEmbeddings;

        var truncated = new Tensor<T>([maxLen, hidDim]);
        for (int s = 0; s < maxLen; s++)
            for (int d = 0; d < hidDim; d++)
                truncated[s, d] = rawEmbeddings[s, d];

        return truncated;
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        if (modelOutput.Rank >= 2 && modelOutput.Shape[^1] == _options.NumLabels)
        {
            return ArgmaxDecode(modelOutput);
        }

        return modelOutput;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? $"{_modelName}-Native" : $"{_modelName}-ONNX",
            Description = $"{_modelName} span-based NER ({_paperCitation})",
            Complexity = _options.NumTransformerLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["HiddenDimension"] = _options.HiddenDimension.ToString();
        m.AdditionalInfo["MaxSpanLength"] = _options.MaxSpanLength.ToString();
        m.AdditionalInfo["SpanEmbeddingDimension"] = _options.SpanEmbeddingDimension.ToString();
        m.AdditionalInfo["NumLabels"] = _options.NumLabels.ToString();
        return m;
    }

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write((int)_options.Variant);
        w.Write(_options.HiddenDimension);
        w.Write(_options.NumAttentionHeads);
        w.Write(_options.NumTransformerLayers);
        w.Write(_options.IntermediateDimension);
        w.Write(_options.NumLabels);
        w.Write(_options.MaxSequenceLength);
        w.Write(_options.MaxSpanLength);
        w.Write(_options.SpanEmbeddingDimension);
        w.Write(_options.DropoutRate);
        w.Write(_options.LearningRate);
        w.Write(_options.NegativeSpanSampleRatio);
        w.Write(_options.LabelNames.Length);
        foreach (var label in _options.LabelNames)
            w.Write(label);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.Variant = (NERModelVariant)r.ReadInt32();
        _options.HiddenDimension = r.ReadInt32();
        _options.NumAttentionHeads = r.ReadInt32();
        _options.NumTransformerLayers = r.ReadInt32();
        _options.IntermediateDimension = r.ReadInt32();
        _options.NumLabels = r.ReadInt32();
        _options.MaxSequenceLength = r.ReadInt32();
        _options.MaxSpanLength = r.ReadInt32();
        _options.SpanEmbeddingDimension = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        _options.LearningRate = r.ReadDouble();
        _options.NegativeSpanSampleRatio = r.ReadInt32();
        int labelCount = r.ReadInt32();
        _options.LabelNames = new string[labelCount];
        for (int i = 0; i < labelCount; i++)
            _options.LabelNames[i] = r.ReadString();

        ApplyOptionsToBase();

        // Native-mode layers (with their trained weights) are already reconstructed by
        // the base DeserializeInternalUnchecked before this override runs, so do NOT
        // clear + re-initialize them here — that would discard the deserialized weights
        // and leave the model randomly initialized. Only an ONNX session needs rebuilding.
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
        {
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
        }
    }

    #endregion

    #region Disposal

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                if (_optimizer is IDisposable disposableOptimizer)
                    disposableOptimizer.Dispose();
            }
            _disposed = true;
        }
        base.Dispose(disposing);
    }

    #endregion

    #region Private Helpers

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? _modelName);
    }

    private void ValidateOptions()
    {
        if (_options.NumLabels != _options.LabelNames.Length)
            throw new ArgumentException(
                $"NumLabels ({_options.NumLabels}) must match LabelNames length ({_options.LabelNames.Length}).");

        if (_options.NumAttentionHeads <= 0)
            throw new ArgumentException(
                $"NumAttentionHeads must be positive. Got: {_options.NumAttentionHeads}.");

        if (_options.HiddenDimension % _options.NumAttentionHeads != 0)
            throw new ArgumentException(
                $"HiddenDimension ({_options.HiddenDimension}) must be divisible by NumAttentionHeads ({_options.NumAttentionHeads}).");
    }

    private void ApplyOptionsToBase()
    {
        NumLabels = _options.NumLabels;
        EmbeddingDimension = _options.HiddenDimension;
        MaxSequenceLength = _options.MaxSequenceLength;
        UseCRF = false; // Span-based models don't use CRF
        LabelNames = _options.LabelNames;
    }

    #endregion
}
