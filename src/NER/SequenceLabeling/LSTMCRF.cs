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

namespace AiDotNet.NER.SequenceLabeling;

/// <summary>
/// LSTM-CRF: Unidirectional LSTM with Conditional Random Field for Named Entity Recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// LSTM-CRF (Huang, Xu, and Yu, 2015 - "Bidirectional LSTM-CRF Models for Sequence Tagging")
/// is a simpler variant of BiLSTM-CRF that uses a unidirectional (left-to-right) LSTM encoder.
/// While the original paper proposed both unidirectional and bidirectional variants, this class
/// implements the unidirectional version for scenarios requiring lower latency or streaming inference.
///
/// <b>Architecture:</b>
/// <code>
///   [word embeddings] --&gt; [LSTM (left-to-right)] --&gt; [Dense projection] --&gt; [CRF decoding] --&gt; [labels]
/// </code>
///
/// <b>Key Differences from BiLSTM-CRF:</b>
/// - Only processes text left-to-right (no backward pass)
/// - Each token's representation only captures preceding context (not following context)
/// - ~50% fewer LSTM parameters (one direction instead of two)
/// - ~2x faster inference per token
/// - 1-2% lower F1 score compared to BiLSTM-CRF
///
/// <b>When to Use LSTM-CRF vs BiLSTM-CRF:</b>
/// - <b>LSTM-CRF:</b> Real-time/streaming NER, edge deployment, latency-sensitive applications
/// - <b>BiLSTM-CRF:</b> Offline processing, maximum accuracy, standard NER benchmarks
///
/// The CRF layer is particularly important for LSTM-CRF because it partially compensates for the
/// lack of right-context. Without the CRF, the unidirectional LSTM would have no way to enforce
/// constraints like "I-PER can only follow B-PER or I-PER", which depend on future label decisions.
/// The CRF's transition matrix captures these patterns, effectively providing a form of right-context
/// at the label level.
/// </para>
/// <para>
/// <b>For Beginners:</b> LSTM-CRF is a faster but slightly less accurate version of BiLSTM-CRF.
/// Instead of reading the sentence both forwards and backwards, it only reads forwards (left to right).
///
/// Think of it like reading a mystery novel: BiLSTM-CRF reads the whole book before deciding who
/// the suspects are, while LSTM-CRF identifies suspects as it reads, without looking ahead.
/// Both are good at their job, but having the full picture (BiLSTM) helps with tricky cases.
///
/// Use this model when speed matters more than getting every last bit of accuracy, such as:
/// - Processing live chat messages for entity extraction
/// - Real-time speech-to-text NER (identifying entities as words are spoken)
/// - Edge/mobile deployment where compute is limited
/// </para>
/// </remarks>
public class LSTMCRF<T> : SequenceLabelingNERBase<T>, INERModel<T>
{
    #region Fields

    private readonly LSTMCRFOptions _options;
    private bool _useNativeMode;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    #endregion

    #region INERModel Properties

    /// <inheritdoc />
    public int[] ExpectedInputShape => [_options.MaxSequenceLength, _options.EmbeddingDimension];

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an LSTM-CRF model in ONNX inference mode using a pre-trained model file.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions.</param>
    /// <param name="modelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="options">Optional model configuration. If null, default options are used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a pre-trained .onnx model file.
    /// </para>
    /// </remarks>
    public LSTMCRF(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        LSTMCRFOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new LSTMCRFOptions();
        _options.ModelPath = modelPath;
        _useNativeMode = false;
        ValidateOptions();
        ApplyOptionsToBase();
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
    }

    /// <summary>
    /// Creates an LSTM-CRF model in native training mode with C# layers.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions.</param>
    /// <param name="options">Optional model configuration. If null, default options are used.</param>
    /// <param name="optimizer">Optional optimizer. If null, AdamW with configured learning rate is used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor to create a fresh model for training.
    /// </para>
    /// </remarks>
    public LSTMCRF(
        NeuralNetworkArchitecture<T> architecture,
        LSTMCRFOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new LSTMCRFOptions();
        _useNativeMode = true;
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
        var preprocessed = PreprocessTokens(tokenEmbeddings);
        Tensor<T> output = preprocessed;

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
                $"Expected rank-2 [seqLen, embDim] or rank-3 [batch, seqLen, embDim] tensor. Got rank {input.Rank}.");

        int embDim = input.Rank == 2 ? input.Shape[1] : input.Shape[2];
        if (embDim != _options.EmbeddingDimension)
            throw new ArgumentException(
                $"Embedding dimension mismatch. Expected {_options.EmbeddingDimension}, got {embDim}.");

        int seqDim = input.Rank == 2 ? input.Shape[0] : input.Shape[1];
        if (seqDim > _options.MaxSequenceLength)
            throw new ArgumentException(
                $"Sequence length {seqDim} exceeds maximum {_options.MaxSequenceLength}. " +
                "Input will be truncated during preprocessing, but consider increasing MaxSequenceLength.");
    }

    /// <inheritdoc />
    string INERModel<T>.GetModelSummary()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("=== LSTM-CRF Model Summary ===");
        sb.AppendLine($"Variant: {_options.Variant}");
        sb.AppendLine($"Mode: {(IsOnnxMode ? "ONNX Inference" : "Native Training")}");
        sb.AppendLine($"Embedding Dimension: {_options.EmbeddingDimension}");
        sb.AppendLine($"Hidden Dimension: {_options.HiddenDimension}");
        sb.AppendLine($"LSTM Layers: {_options.NumLSTMLayers}");
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultLSTMCRFLayers(
                embeddingDimension: _options.EmbeddingDimension,
                hiddenDimension: _options.HiddenDimension,
                numLabels: _options.NumLabels,
                numLSTMLayers: _options.NumLSTMLayers,
                maxSequenceLength: _options.MaxSequenceLength,
                dropoutRate: _options.DropoutRate));
        }
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
        int embDim = _options.EmbeddingDimension;

        if (rawEmbeddings.Rank < 2) return rawEmbeddings;

        int seqLen = rawEmbeddings.Shape[0];
        if (seqLen == maxLen) return rawEmbeddings;

        var padded = new Tensor<T>([maxLen, embDim]);
        int copyLen = Math.Min(seqLen, maxLen);
        for (int s = 0; s < copyLen; s++)
            for (int d = 0; d < embDim; d++)
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
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new LSTMCRFOptions(_options);

        if (!_useNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new LSTMCRF<T>(Architecture, p, optionsCopy);

        return new LSTMCRF<T>(Architecture, optionsCopy);
    }

    #endregion

    #region Metadata and Serialization

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "LSTM-CRF-Native" : "LSTM-CRF-ONNX",
            Description = $"LSTM-CRF {_options.Variant} unidirectional sequence labeling NER (Huang et al., 2015)",
            ModelType = ModelType.NamedEntityRecognition,
            Complexity = _options.NumLSTMLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["EmbeddingDimension"] = _options.EmbeddingDimension.ToString();
        m.AdditionalInfo["HiddenDimension"] = _options.HiddenDimension.ToString();
        m.AdditionalInfo["NumLSTMLayers"] = _options.NumLSTMLayers.ToString();
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
        w.Write(_options.EmbeddingDimension);
        w.Write(_options.HiddenDimension);
        w.Write(_options.NumLSTMLayers);
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
        _options.EmbeddingDimension = r.ReadInt32();
        _options.HiddenDimension = r.ReadInt32();
        _options.NumLSTMLayers = r.ReadInt32();
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

    private bool _disposed;

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(LSTMCRF<T>));
    }

    private void ValidateOptions()
    {
        if (_options.NumLabels != _options.LabelNames.Length)
            throw new ArgumentException(
                $"NumLabels ({_options.NumLabels}) must match LabelNames length ({_options.LabelNames.Length}).");
    }

    private void ApplyOptionsToBase()
    {
        NumLabels = _options.NumLabels;
        EmbeddingDimension = _options.EmbeddingDimension;
        MaxSequenceLength = _options.MaxSequenceLength;
        UseCRF = _options.UseCRF;
        LabelNames = _options.LabelNames;
    }

    #endregion
}
