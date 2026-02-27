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
/// CNN-BiLSTM-CRF: Character CNN + Bidirectional LSTM + Conditional Random Field for Named Entity Recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CNN-BiLSTM-CRF (Ma and Hovy, ACL 2016 - "End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF")
/// is a state-of-the-art sequence labeling architecture that extends BiLSTM-CRF with character-level CNN
/// embeddings. The model has three key components:
///
/// <b>1. Character-level CNN:</b>
/// A 1D Convolutional Neural Network that processes each word's character sequence to extract morphological
/// features. Unlike the character BiLSTM used in Lample et al. (2016), a CNN is:
/// - Faster: Processes all character positions in parallel (no sequential dependency)
/// - Better at local patterns: Captures character trigrams/n-grams like "-tion", "Dr.", "co-"
/// - Simpler: Fewer parameters and easier to train
///
/// The CNN operates as follows:
/// (a) Each character is mapped to a 30-dimensional embedding vector
/// (b) A bank of 30 convolutional filters with kernel size 3 slides over the character sequence
/// (c) Max-pooling over the character sequence produces a fixed-size 30-dimensional feature vector
/// (d) This character feature vector is concatenated with the word embedding
///
/// This captures features that word embeddings miss:
/// - Capitalization: "Apple" (entity) vs "apple" (fruit)
/// - Suffixes/prefixes: "-burg" (city), "-son" (person), "Dr." (title), "un-" (negation)
/// - Out-of-vocabulary words: Rare names recognized by character patterns
/// - Number patterns: "2023" recognized as a potential date
///
/// <b>2. Bidirectional LSTM (BiLSTM):</b>
/// Processes the concatenated [word_embedding; char_CNN_features] in both directions.
/// The forward LSTM reads left-to-right and the backward LSTM reads right-to-left.
/// Their hidden states are merged at each position via element-wise addition, giving each
/// token a context-aware representation informed by both preceding and following words.
///
/// <b>3. Conditional Random Field (CRF):</b>
/// Models label-level transition dependencies using a learned transition matrix. During inference,
/// the Viterbi algorithm finds the globally optimal label sequence that maximizes both emission
/// scores (from the BiLSTM) and transition scores (learned label-to-label preferences).
///
/// <b>Performance (CoNLL-2003):</b>
/// - CNN-BiLSTM-CRF achieves 91.21% F1 (Ma and Hovy, 2016)
/// - BiLSTM-CRF achieves 90.94% F1 (Lample et al., 2016)
/// - The character CNN provides a modest but consistent improvement over character BiLSTM
///
/// <b>Architecture diagram:</b>
/// <code>
///   [char embeddings] --&gt; [1D CNN + MaxPool] --&gt; [char features]
///                                                        |
///   [word embeddings] ---------------------------------- + (concatenate)
///                                                        |
///                                                   [BiLSTM layers]
///                                                        |
///                                                   [Dense projection]
///                                                        |
///                                                   [CRF decoding]
///                                                        |
///                                                   [label sequence]
/// </code>
/// </para>
/// <para>
/// <b>For Beginners:</b> CNN-BiLSTM-CRF is one of the best-performing non-transformer NER models.
/// It works in three steps:
///
/// 1. <b>Character CNN:</b> Looks at the letters within each word to detect patterns. For example,
///    it can notice that "Google" starts with a capital letter and "Inc." ends with a period,
///    which are clues that these might be parts of a company name.
///
/// 2. <b>BiLSTM:</b> Reads the sentence both forwards and backwards to understand each word in
///    context. For example, "Apple" after "bought" is likely a company, but "Apple" after "ate" is
///    likely a fruit.
///
/// 3. <b>CRF:</b> Makes sure the final labels are consistent. For example, if one word is labeled
///    as the start of a person name (B-PER), the next word should be either a continuation (I-PER)
///    or a new entity/non-entity, not the start of an organization.
///
/// This model is a good choice when:
/// - You need high accuracy without transformer-level compute costs
/// - Your text contains many rare or out-of-vocabulary words (names, technical terms)
/// - You need fast training and inference compared to BERT-based models
/// </para>
/// </remarks>
public class CNNBiLSTMCRF<T> : SequenceLabelingNERBase<T>, INERModel<T>
{
    #region Fields

    private readonly CNNBiLSTMCRFOptions _options;
    private bool _useNativeMode;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    #endregion

    #region INERModel Properties

    /// <inheritdoc />
    public int[] ExpectedInputShape => [_options.MaxSequenceLength, _options.EmbeddingDimension];

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a CNN-BiLSTM-CRF model in ONNX inference mode using a pre-trained model file.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions.</param>
    /// <param name="modelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="options">Optional model configuration. If null, default options are used.</param>
    /// <remarks>
    /// <para>
    /// In ONNX mode, the model loads pre-trained weights from the specified file for fast inference.
    /// Training is not supported in this mode. Use this constructor when you have a model trained
    /// in PyTorch or TensorFlow and exported to ONNX format.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a pre-trained .onnx model file.
    /// The model will be ready for predictions immediately without any training.
    /// </para>
    /// </remarks>
    public CNNBiLSTMCRF(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        CNNBiLSTMCRFOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new CNNBiLSTMCRFOptions();
        _options.ModelPath = modelPath;
        _useNativeMode = false;
        ValidateOptions();
        ApplyOptionsToBase();
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
    }

    /// <summary>
    /// Creates a CNN-BiLSTM-CRF model in native training mode with C# layers.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions.</param>
    /// <param name="options">Optional model configuration. If null, default options are used.</param>
    /// <param name="optimizer">Optional optimizer for training. If null, AdamW with the configured
    /// learning rate is used.</param>
    /// <remarks>
    /// <para>
    /// In native mode, the model builds its layers using the LayerHelper and can be trained
    /// from scratch on labeled NER data. This is the constructor to use when you want to
    /// train a new model on your own dataset.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor to create a fresh model that you'll train
    /// on your own labeled data. The model starts with random weights and learns from examples.
    /// </para>
    /// </remarks>
    public CNNBiLSTMCRF(
        NeuralNetworkArchitecture<T> architecture,
        CNNBiLSTMCRFOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new CNNBiLSTMCRFOptions();
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
        ThrowIfDisposed();
        var preprocessed = PreprocessTokens(tokenEmbeddings);

        if (IsOnnxMode)
        {
            return RunOnnxInference(preprocessed);
        }

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
                    var grad = LossFunction.CalculateDerivative(output.ToVector(), preprocessedLabels.ToVector());
                    var gt = Tensor<T>.FromVector(grad);
                    for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt);
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
        sb.AppendLine("=== CNN-BiLSTM-CRF Model Summary ===");
        sb.AppendLine($"Variant: {_options.Variant}");
        sb.AppendLine($"Mode: {(IsOnnxMode ? "ONNX Inference" : "Native Training")}");
        sb.AppendLine($"Embedding Dimension: {_options.EmbeddingDimension}");
        sb.AppendLine($"Hidden Dimension: {_options.HiddenDimension}");
        sb.AppendLine($"LSTM Layers: {_options.NumLSTMLayers}");
        sb.AppendLine($"Char CNN Filters: {_options.CharCNNFilters}");
        sb.AppendLine($"Char CNN Kernel Size: {_options.CharCNNKernelSize}");
        sb.AppendLine($"Char Embedding Dim: {_options.CharEmbeddingDimension}");
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultCNNBiLSTMCRFLayers(
                embeddingDimension: _options.EmbeddingDimension,
                hiddenDimension: _options.HiddenDimension,
                numLabels: _options.NumLabels,
                numLSTMLayers: _options.NumLSTMLayers,
                maxSequenceLength: _options.MaxSequenceLength,
                dropoutRate: _options.DropoutRate,
                charEmbeddingDimension: _options.CharEmbeddingDimension,
                charCNNFilters: _options.CharCNNFilters,
                charCNNKernelSize: _options.CharCNNKernelSize));
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
            var preprocessedLabels = PreprocessLabels(expected, preprocessed.Shape[0]);
            var output = Forward(preprocessed);
            var grad = LossFunction.CalculateDerivative(output.ToVector(), preprocessedLabels.ToVector());
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

        // Handle rank-3 [batch, seqLen, embDim]
        if (rawEmbeddings.Rank == 3)
        {
            int batch = rawEmbeddings.Shape[0];
            int seqLen3 = rawEmbeddings.Shape[1];
            if (seqLen3 == maxLen) return rawEmbeddings;

            var padded3 = new Tensor<T>([batch, maxLen, embDim]);
            int copyLen3 = Math.Min(seqLen3, maxLen);
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < copyLen3; s++)
                    for (int d = 0; d < embDim; d++)
                        padded3[b, s, d] = rawEmbeddings[b, s, d];
            return padded3;
        }

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
        var optionsCopy = new CNNBiLSTMCRFOptions(_options);

        if (!_useNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new CNNBiLSTMCRF<T>(Architecture, p, optionsCopy);

        return new CNNBiLSTMCRF<T>(Architecture, optionsCopy);
    }

    #endregion

    #region Metadata and Serialization

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "CNN-BiLSTM-CRF-Native" : "CNN-BiLSTM-CRF-ONNX",
            Description = $"CNN-BiLSTM-CRF {_options.Variant} sequence labeling NER (Ma and Hovy, ACL 2016)",
            ModelType = ModelType.NamedEntityRecognition,
            Complexity = _options.NumLSTMLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["EmbeddingDimension"] = _options.EmbeddingDimension.ToString();
        m.AdditionalInfo["HiddenDimension"] = _options.HiddenDimension.ToString();
        m.AdditionalInfo["NumLSTMLayers"] = _options.NumLSTMLayers.ToString();
        m.AdditionalInfo["NumLabels"] = _options.NumLabels.ToString();
        m.AdditionalInfo["CharCNNFilters"] = _options.CharCNNFilters.ToString();
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
        w.Write(_options.CharEmbeddingDimension);
        w.Write(_options.CharCNNFilters);
        w.Write(_options.CharCNNKernelSize);
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
        _options.CharEmbeddingDimension = r.ReadInt32();
        _options.CharCNNFilters = r.ReadInt32();
        _options.CharCNNKernelSize = r.ReadInt32();
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
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(CNNBiLSTMCRF<T>));
    }

    private void ValidateOptions()
    {
        if (_options.NumLabels != _options.LabelNames.Length)
            throw new ArgumentException(
                $"NumLabels ({_options.NumLabels}) must match LabelNames length ({_options.LabelNames.Length}).");
    }

    private Tensor<T> PreprocessLabels(Tensor<T> labels, int targetSeqLen)
    {
        if (labels.Rank < 1) return labels;

        int labelLen = labels.Shape[0];
        if (labelLen == targetSeqLen) return labels;

        if (labels.Rank == 1)
        {
            var padded = new Tensor<T>([targetSeqLen]);
            int copyLen = Math.Min(labelLen, targetSeqLen);
            for (int i = 0; i < copyLen; i++) padded[i] = labels[i];
            return padded;
        }

        int cols = labels.Shape[1];
        var padded2 = new Tensor<T>([targetSeqLen, cols]);
        int copyLen2 = Math.Min(labelLen, targetSeqLen);
        for (int s = 0; s < copyLen2; s++)
            for (int c = 0; c < cols; c++)
                padded2[s, c] = labels[s, c];
        return padded2;
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
