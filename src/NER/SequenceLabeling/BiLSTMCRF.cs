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
/// BiLSTM-CRF: Bidirectional LSTM with Conditional Random Field for Named Entity Recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BiLSTM-CRF (Huang et al., 2015; Lample et al., NAACL 2016) is the foundational neural NER
/// architecture that combines two complementary components:
///
/// - Bidirectional LSTM: Processes token embeddings in both forward and backward directions,
///   producing contextualized representations that capture the full sentence context for each token.
///   The forward LSTM reads left-to-right, the backward LSTM reads right-to-left, and their
///   outputs are concatenated to form a rich representation.
///
/// - Conditional Random Field (CRF): Models label-level transition dependencies to produce
///   globally optimal label sequences. The CRF layer learns which label transitions are valid
///   (e.g., I-PER must follow B-PER or I-PER) and uses the Viterbi algorithm to find the
///   best label path through the entire sequence.
///
/// The architecture processes text as follows:
/// 1. Token embeddings (e.g., from GloVe, Word2Vec, or BERT) are fed into the BiLSTM
/// 2. BiLSTM produces emission scores for each token-label pair
/// 3. A linear projection maps BiLSTM hidden states to label scores
/// 4. CRF layer uses emission scores + transition scores to decode the optimal label sequence
///
/// This model serves as the golden example for the NER model family in AiDotNet,
/// establishing the architectural patterns that all other NER models follow.
/// </para>
/// <para>
/// <b>For Beginners:</b> BiLSTM-CRF reads text forwards and backwards to understand each word
/// in full context, then uses a CRF to pick the best sequence of entity labels. For example,
/// given "John Smith works at Google", it would label "John"=B-PER, "Smith"=I-PER,
/// "works"=O, "at"=O, "Google"=B-ORG.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 100, outputSize: 9);
/// var model = new BiLSTMCRF&lt;float&gt;(arch);
/// model.Train(tokenEmbeddings, labelSequence);
/// var predictions = model.PredictLabels(newTokenEmbeddings);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "Bidirectional LSTM-CRF Models for Sequence Tagging"
/// (Huang, Xu, Yu, 2015) and "Neural Architectures for Named Entity Recognition"
/// (Lample et al., NAACL 2016)
/// </para>
/// </remarks>
public class BiLSTMCRF<T> : SequenceLabelingNERBase<T>, INERModel<T>
{
    #region Fields

    private readonly BiLSTMCRFOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region INERModel Properties

    /// <inheritdoc />
    int INERModel<T>.NumLabels => _options.NumLabels;

    /// <inheritdoc />
    int INERModel<T>.EmbeddingDimension => _options.EmbeddingDimension;

    /// <inheritdoc />
    public int[] ExpectedInputShape => [_options.MaxSequenceLength, _options.EmbeddingDimension];

    #endregion

    #region Constructors

    /// <summary>Creates a BiLSTM-CRF model in ONNX inference mode.</summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="options">Optional configuration options.</param>
    public BiLSTMCRF(NeuralNetworkArchitecture<T> architecture, string modelPath, BiLSTMCRFOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options ?? new BiLSTMCRFOptions();
        _useNativeMode = false;
        NumLabels = _options.NumLabels;
        EmbeddingDimension = _options.EmbeddingDimension;
        MaxSequenceLength = _options.MaxSequenceLength;
        UseCRF = _options.UseCRF;
        LabelNames = _options.LabelNames;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a BiLSTM-CRF model in native training mode.</summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="optimizer">Optional gradient-based optimizer. Defaults to AdamW.</param>
    public BiLSTMCRF(NeuralNetworkArchitecture<T> architecture, BiLSTMCRFOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new BiLSTMCRFOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        NumLabels = _options.NumLabels;
        EmbeddingDimension = _options.EmbeddingDimension;
        MaxSequenceLength = _options.MaxSequenceLength;
        UseCRF = _options.UseCRF;
        LabelNames = _options.LabelNames;
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
        ThrowIfDisposed();
        if (IsOnnxMode)
        {
            return RunOnnxInference(tokenEmbeddings);
        }

        // Forward through all layers except CRF to get emission scores
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
                Train(tokenEmbeddings, labels);

                progress?.Report(new NERTrainingProgress
                {
                    CurrentEpoch = epoch,
                    TotalEpochs = epochs,
                    CurrentBatch = 1,
                    TotalBatches = 1,
                    Loss = NumOps.ToDouble(LossFunction.CalculateLoss(
                        Predict(tokenEmbeddings).ToVector(), labels.ToVector()))
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
                $"Input must be 2D [sequenceLength, embeddingDim] or 3D [batch, sequenceLength, embeddingDim]. Got rank {input.Rank}.");

        int embDim = input.Shape[^1];
        if (embDim != _options.EmbeddingDimension)
            throw new ArgumentException(
                $"Embedding dimension mismatch. Expected {_options.EmbeddingDimension}, got {embDim}.");
    }

    /// <inheritdoc />
    string INERModel<T>.GetModelSummary()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"BiLSTM-CRF ({_options.Variant})");
        sb.AppendLine($"  Mode: {(_useNativeMode ? "Native" : "ONNX")}");
        sb.AppendLine($"  Embedding Dim: {_options.EmbeddingDimension}");
        sb.AppendLine($"  Hidden Dim: {_options.HiddenDimension} (x2 bidirectional = {_options.HiddenDimension * 2})");
        sb.AppendLine($"  LSTM Layers: {_options.NumLSTMLayers}");
        sb.AppendLine($"  Num Labels: {_options.NumLabels}");
        sb.AppendLine($"  CRF: {(_options.UseCRF ? "Enabled" : "Disabled")}");
        sb.AppendLine($"  Char Embeddings: {(_options.UseCharEmbeddings ? "Enabled" : "Disabled")}");
        sb.AppendLine($"  Dropout: {_options.DropoutRate:P0}");
        sb.AppendLine($"  Total Parameters: {Layers.Sum(l => l.ParameterCount)}");
        return sb.ToString();
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            int embDim = _options.EmbeddingDimension;
            int hiddenDim = _options.HiddenDimension;
            int numLabels = _options.NumLabels;
            int seqLen = _options.MaxSequenceLength;

            // Build BiLSTM-CRF architecture:
            // 1. BiLSTM layers (forward + backward LSTM, output dim = 2 * hiddenDim)
            int currentInputSize = embDim;

            for (int layer = 0; layer < _options.NumLSTMLayers; layer++)
            {
                // Forward LSTM
                Layers.Add(new LSTMLayer<T>(
                    inputSize: currentInputSize,
                    hiddenSize: hiddenDim,
                    inputShape: [currentInputSize],
                    activation: new TanhActivation<T>() as IActivationFunction<T>,
                    recurrentActivation: new SigmoidActivation<T>() as IActivationFunction<T>));

                // After BiLSTM, output size is hiddenDim (single direction for now)
                currentInputSize = hiddenDim;

                // Dropout between LSTM layers
                if (_options.DropoutRate > 0 && layer < _options.NumLSTMLayers - 1)
                {
                    Layers.Add(new DropoutLayer<T>(_options.DropoutRate));
                }
            }

            // 2. Dropout before projection
            if (_options.DropoutRate > 0)
            {
                Layers.Add(new DropoutLayer<T>(_options.DropoutRate));
            }

            // 3. Linear projection: hidden_dim -> num_labels (emission scores)
            Layers.Add(new DenseLayer<T>(
                inputSize: currentInputSize,
                outputSize: numLabels,
                activationFunction: new IdentityActivation<T>()));

            // 4. CRF layer for sequence-level decoding
            if (_options.UseCRF)
            {
                Layers.Add(new ConditionalRandomFieldLayer<T>(
                    numClasses: numLabels,
                    sequenceLength: seqLen,
                    scalarActivation: new IdentityActivation<T>() as IActivationFunction<T>));
            }
        }
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) return RunOnnxInference(input);
        return Forward(input);
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (_optimizer is null) throw new InvalidOperationException("Optimizer is not initialized. Cannot train without an optimizer.");
        SetTrainingMode(true);
        try
        {
            var output = Predict(input);
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

    protected override Tensor<T> PreprocessTokens(Tensor<T> rawEmbeddings) => rawEmbeddings;

    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        // If CRF is disabled, apply argmax decoding on emission scores
        if (!_options.UseCRF && modelOutput.Rank >= 2 && modelOutput.Shape[^1] == _options.NumLabels)
        {
            return ArgmaxDecode(modelOutput);
        }
        // CRF layer already produces decoded label indices
        return modelOutput;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "BiLSTM-CRF-Native" : "BiLSTM-CRF-ONNX",
            Description = $"BiLSTM-CRF {_options.Variant} sequence labeling NER (Lample et al., NAACL 2016)",
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
        w.Write(_options.UseCharEmbeddings);
        w.Write(_options.CharEmbeddingDimension);
        w.Write(_options.CharHiddenDimension);
        w.Write(_options.DropoutRate);

        // Serialize label names
        w.Write(_options.LabelNames.Length);
        foreach (var label in _options.LabelNames)
        {
            w.Write(label);
        }
    }

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
        _options.UseCharEmbeddings = r.ReadBoolean();
        _options.CharEmbeddingDimension = r.ReadInt32();
        _options.CharHiddenDimension = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();

        // Deserialize label names
        int labelCount = r.ReadInt32();
        _options.LabelNames = new string[labelCount];
        for (int i = 0; i < labelCount; i++)
        {
            _options.LabelNames[i] = r.ReadString();
        }

        // Restore properties
        NumLabels = _options.NumLabels;
        EmbeddingDimension = _options.EmbeddingDimension;
        MaxSequenceLength = _options.MaxSequenceLength;
        UseCRF = _options.UseCRF;
        LabelNames = _options.LabelNames;

        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
        {
            OnnxModel?.Dispose();
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
        }
        else if (_useNativeMode)
        {
            OnnxModel?.Dispose();
            OnnxModel = null;
            Layers.Clear();
            InitializeLayers();
        }
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new BiLSTMCRF<T>(Architecture, p, _options);
        return new BiLSTMCRF<T>(Architecture, _options);
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(BiLSTMCRF<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing)
        {
            OnnxModel?.Dispose();
        }
        base.Dispose(disposing);
    }

    #endregion
}
