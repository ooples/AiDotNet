using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NER.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.NER.TransformerBased;

/// <summary>
/// Base class for transformer-based NER models (BERT-NER, RoBERTa-NER, DeBERTa-NER, etc.).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// All transformer-based NER models share the same fine-tuning architecture:
///
/// <code>
///   [token embeddings] --&gt; [Transformer Encoder (N layers)] --&gt; [Dropout] --&gt; [Dense: hidden -&gt; labels] --&gt; [CRF?] --&gt; [labels]
/// </code>
///
/// The transformer encoder consists of N stacked layers, each containing:
/// 1. Multi-head self-attention: Each token attends to all other tokens
/// 2. Feed-forward network: Non-linear transformation of each token independently
/// 3. Layer normalization and residual connections: Stabilize training
///
/// The key differences between BERT, RoBERTa, DeBERTa, etc. are in the pre-training
/// (masked language model, replaced token detection, etc.) and attention mechanism
/// (absolute vs relative position encoding, disentangled attention, etc.). The NER
/// fine-tuning architecture is identical across all variants.
///
/// This base class provides the common implementation. Derived classes only need to:
/// 1. Pass their model-specific name to the base constructor
/// 2. Optionally override CreateDefaultLayers for architecture-specific layer configurations
/// </para>
/// <para>
/// <b>For Beginners:</b> Transformer models are the most accurate NER models available.
/// They use "self-attention" to understand each word by looking at all other words in the
/// sentence simultaneously. Different transformer variants (BERT, RoBERTa, etc.) are like
/// different editions of the same textbook - they have the same structure but were trained
/// slightly differently, leading to different strengths.
/// </para>
/// </remarks>
public abstract class TransformerNERBase<T> : SequenceLabeling.SequenceLabelingNERBase<T>, INERModel<T>
{
    #region Fields

    private readonly TransformerNEROptions _options;
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
    /// Creates a transformer NER model in ONNX inference mode.
    /// </summary>
    protected TransformerNERBase(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        TransformerNEROptions options,
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
    /// Creates a transformer NER model in native training mode.
    /// </summary>
    protected TransformerNERBase(
        NeuralNetworkArchitecture<T> architecture,
        TransformerNEROptions options,
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

        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate
            });

        InitializeLayers();
    }

    #endregion

    #region Properties

    /// <summary>
    /// Gets the options for this transformer NER model.
    /// </summary>
    protected TransformerNEROptions NEROptions => _options;

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
        return Task.Run(() =>
        {
            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                SetTrainingMode(true);
                var output = Forward(PreprocessTokens(tokenEmbeddings));
                double loss = NumOps.ToDouble(LossFunction.CalculateLoss(
                    output.ToVector(), labels.ToVector()));
                SetTrainingMode(false);

                Train(tokenEmbeddings, labels);

                progress?.Report(new NERTrainingProgress
                {
                    CurrentEpoch = epoch,
                    TotalEpochs = epochs,
                    CurrentBatch = 1,
                    TotalBatches = 1,
                    Loss = loss
                });
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
        sb.AppendLine($"Variant: {_options.Variant}");
        sb.AppendLine($"Mode: {(IsOnnxMode ? "ONNX Inference" : "Native Training")}");
        sb.AppendLine($"Hidden Dimension: {_options.HiddenDimension}");
        sb.AppendLine($"Attention Heads: {_options.NumAttentionHeads}");
        sb.AppendLine($"Transformer Layers: {_options.NumTransformerLayers}");
        sb.AppendLine($"Intermediate Dimension: {_options.IntermediateDimension}");
        sb.AppendLine($"Num Labels: {_options.NumLabels}");
        sb.AppendLine($"Max Sequence Length: {_options.MaxSequenceLength}");
        sb.AppendLine($"Use CRF: {_options.UseCRF}");
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
    /// Creates the default layer stack for this transformer NER model.
    /// </summary>
    /// <returns>Ordered sequence of layers.</returns>
    /// <remarks>
    /// <para>
    /// Override this in derived classes to customize the layer architecture.
    /// The default implementation uses TransformerEncoderLayer + Dense + optional CRF.
    /// </para>
    /// </remarks>
    protected virtual IEnumerable<ILayer<T>> CreateDefaultLayers()
    {
        return LayerHelper<T>.CreateDefaultTransformerNERLayers(
            hiddenDimension: _options.HiddenDimension,
            numAttentionHeads: _options.NumAttentionHeads,
            numTransformerLayers: _options.NumTransformerLayers,
            intermediateDimension: _options.IntermediateDimension,
            numLabels: _options.NumLabels,
            maxSequenceLength: _options.MaxSequenceLength,
            dropoutRate: _options.DropoutRate,
            useCRF: _options.UseCRF);
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (_optimizer is null) throw new InvalidOperationException("Optimizer is not initialized.");
        SetTrainingMode(true);
        try
        {
            var preprocessed = PreprocessTokens(input);
            var output = Forward(preprocessed);
            var grad = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
            var gt = Tensor<T>.FromVector(grad);
            for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt);
            _optimizer.UpdateParameters(Layers);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("Parameter updates are not supported in ONNX mode.");
        int idx = 0;
        foreach (var layer in Layers)
        {
            int count = layer.ParameterCount;
            layer.UpdateParameters(parameters.Slice(idx, count));
            idx += count;
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> PreprocessTokens(Tensor<T> rawEmbeddings)
    {
        int maxLen = _options.MaxSequenceLength;
        int hidDim = _options.HiddenDimension;

        if (rawEmbeddings.Rank < 2) return rawEmbeddings;

        int seqLen = rawEmbeddings.Shape[0];
        if (seqLen == maxLen) return rawEmbeddings;

        var padded = new Tensor<T>([maxLen, hidDim]);
        int copyLen = Math.Min(seqLen, maxLen);
        for (int s = 0; s < copyLen; s++)
            for (int d = 0; d < hidDim; d++)
                padded[s, d] = rawEmbeddings[s, d];

        return padded;
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
            Description = $"{_modelName} {_options.Variant} transformer NER ({_paperCitation})",
            ModelType = ModelType.NamedEntityRecognition,
            Complexity = _options.NumTransformerLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["HiddenDimension"] = _options.HiddenDimension.ToString();
        m.AdditionalInfo["NumAttentionHeads"] = _options.NumAttentionHeads.ToString();
        m.AdditionalInfo["NumTransformerLayers"] = _options.NumTransformerLayers.ToString();
        m.AdditionalInfo["NumLabels"] = _options.NumLabels.ToString();
        m.AdditionalInfo["UseCRF"] = _options.UseCRF.ToString();
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
        w.Write(_options.UseCRF);
        w.Write(_options.DropoutRate);
        w.Write(_options.LearningRate);
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
        _options.UseCRF = r.ReadBoolean();
        _options.DropoutRate = r.ReadDouble();
        _options.LearningRate = r.ReadDouble();
        int labelCount = r.ReadInt32();
        _options.LabelNames = new string[labelCount];
        for (int i = 0; i < labelCount; i++)
            _options.LabelNames[i] = r.ReadString();

        ApplyOptionsToBase();

        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
        {
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
        }
        else if (_useNativeMode)
        {
            Layers.Clear();
            InitializeLayers();
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

        if (_options.HiddenDimension % _options.NumAttentionHeads != 0)
            throw new ArgumentException(
                $"HiddenDimension ({_options.HiddenDimension}) must be divisible by NumAttentionHeads ({_options.NumAttentionHeads}).");
    }

    private void ApplyOptionsToBase()
    {
        NumLabels = _options.NumLabels;
        EmbeddingDimension = _options.HiddenDimension;
        MaxSequenceLength = _options.MaxSequenceLength;
        UseCRF = _options.UseCRF;
        LabelNames = _options.LabelNames;
    }

    #endregion
}
