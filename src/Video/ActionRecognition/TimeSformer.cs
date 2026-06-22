using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.ActionRecognition;

/// <summary>
/// TimeSformer: Is Space-Time Attention All You Need for Video Understanding?
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> TimeSformer is a transformer-based model for video classification
/// that applies attention across both space and time dimensions. Unlike CNNs that use
/// 3D convolutions, TimeSformer uses pure self-attention to understand video content.
///
/// Key capabilities:
/// - Video action recognition (classify what action is happening)
/// - Temporal reasoning (understand events across time)
/// - Scene understanding (understand spatial context)
///
/// The model uses "divided space-time attention" where:
/// 1. First, attention is applied across time (same spatial location, different frames)
/// 2. Then, attention is applied across space (same frame, different locations)
///
/// Example usage (native mode for training):
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3);
/// var model = new TimeSformer&lt;double&gt;(arch, numClasses: 400);
/// model.Train(videoFrames, expectedLabels);
/// var predictions = model.Classify(videoFrames);
/// </code>
///
/// Example usage (ONNX mode for inference only):
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3);
/// var model = new TimeSformer&lt;double&gt;(arch, "timesformer.onnx");
/// var predictions = model.Classify(videoFrames);
/// </code>
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Divided space-time attention for efficiency
/// - Patch embedding similar to ViT
/// - Learnable positional embeddings for space and time
/// - Classification token for final prediction
/// </para>
/// <para>
/// <b>Reference:</b> "Is Space-Time Attention All You Need for Video Understanding?"
/// https://arxiv.org/abs/2102.05095
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Is Space-Time Attention All You Need for Video Understanding?",
    "https://arxiv.org/abs/2102.05095",
    Year = 2021,
    Authors = "Gedas Bertasius, Heng Wang, Lorenzo Torresani")]
public class TimeSformer<T> : NeuralNetworkBase<T>
{
    private readonly TimeSformerOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Execution Mode

    private readonly bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    private readonly InferenceSession? _onnxSession;
    private readonly string? _onnxModelPath;

    #endregion

    #region Native Mode Fields

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly int _embedDim;
    private readonly int _numHeads;
    private readonly int _numLayers;
    private readonly int _numFrames;
    private readonly int _patchSize;
    private readonly int _imageSize;
    private readonly int _numClasses;
    private readonly AttentionType _attentionType;
    private bool _usesTokenizedNativePath;

    #endregion

    #region Properties

    /// <summary>
    /// Gets whether this model uses native mode (true) or ONNX mode (false).
    /// </summary>
    internal bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets whether training is supported (only in native mode).
    /// </summary>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    internal int EmbedDim => _embedDim;

    /// <summary>
    /// Gets the number of frames processed.
    /// </summary>
    internal int NumFrames => _numFrames;

    /// <summary>
    /// Gets the number of output classes.
    /// </summary>
    internal int NumClasses => _numClasses;

    /// <summary>
    /// Gets the attention type used.
    /// </summary>
    internal AttentionType AttentionMode => _attentionType;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a TimeSformer model using native layers for training and inference.
    /// </summary>
    /// <param name="architecture">Architecture for the video encoder.</param>
    /// <param name="numClasses">Number of output classes for classification.</param>
    /// <param name="optimizer">Optional optimizer for training. Default: Adam.</param>
    /// <param name="lossFunction">Optional loss function. Default: CrossEntropy.</param>
    /// <param name="embedDim">Embedding dimension (default: 768).</param>
    /// <param name="numHeads">Number of attention heads (default: 12).</param>
    /// <param name="numLayers">Number of transformer layers (default: 12).</param>
    /// <param name="numFrames">Number of frames to process (default: 8).</param>
    /// <param name="patchSize">Patch size for tokenization (default: 16).</param>
    /// <param name="attentionType">Type of space-time attention (default: DividedSpaceTime).</param>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public TimeSformer()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.ThreeDimensional,
            taskType: Enums.NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 224, inputWidth: 224, inputDepth: 3,
            outputSize: 400))
    {
    }

    public TimeSformer(
        NeuralNetworkArchitecture<T> architecture,
        int numClasses = 400,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int embedDim = 768,
        int numHeads = 12,
        int numLayers = 12,
        int numFrames = 8,
        int patchSize = 16,
        AttentionType attentionType = AttentionType.DividedSpaceTime,
        TimeSformerOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new TimeSformerOptions();
        Options = _options;

        if (embedDim < 1)
            throw new ArgumentOutOfRangeException(nameof(embedDim), "Embedding dimension must be at least 1.");
        if (numLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numLayers), "Number of layers must be at least 1.");
        if (numHeads < 1)
            throw new ArgumentOutOfRangeException(nameof(numHeads), "Number of heads must be at least 1.");
        if (embedDim % numHeads != 0)
            throw new ArgumentException("Embedding dimension must be divisible by the number of attention heads.", nameof(numHeads));
        if (patchSize < 1)
            throw new ArgumentOutOfRangeException(nameof(patchSize), "Patch size must be at least 1.");
        if (numClasses < 1)
            throw new ArgumentOutOfRangeException(nameof(numClasses), "Number of classes must be at least 1.");

        _useNativeMode = true;
        _embedDim = embedDim;
        _numHeads = numHeads;
        _numLayers = numLayers;
        _numFrames = numFrames;
        _patchSize = patchSize;
        _imageSize = architecture.InputHeight > 0 ? architecture.InputHeight : 224;
        _numClasses = numClasses;
        _attentionType = attentionType;

        _lossFunction = lossFunction ?? new CrossEntropyWithLogitsLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a TimeSformer model using a pretrained ONNX model for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the pretrained ONNX model.</param>
    /// <param name="numClasses">Number of output classes (default: 400 for Kinetics).</param>
    /// <param name="embedDim">Embedding dimension of the model (default: 768).</param>
    public TimeSformer(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 400,
        int embedDim = 768,
        TimeSformerOptions? options = null)
        : base(architecture, new CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new TimeSformerOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"TimeSformer ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _embedDim = embedDim;
        _numHeads = 12;
        _numLayers = 12;
        _numFrames = 8;
        _patchSize = 16;
        _imageSize = architecture.InputHeight > 0 ? architecture.InputHeight : 224;
        _numClasses = numClasses;
        _attentionType = AttentionType.DividedSpaceTime;
        _lossFunction = new CrossEntropyWithLogitsLoss<T>();

        try
        {
            _onnxSession = new InferenceSession(onnxModelPath);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load ONNX model: {ex.Message}", ex);
        }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Classifies video frames into action categories.
    /// </summary>
    /// <param name="videoFrames">Video frames tensor [B, T, C, H, W] or [T, C, H, W].</param>
    /// <returns>Class probabilities tensor [B, NumClasses] or [NumClasses].</returns>
    public Tensor<T> Classify(Tensor<T> videoFrames)
    {
        if (videoFrames is null)
            throw new ArgumentNullException(nameof(videoFrames));

        if (_useNativeMode)
        {
            return Forward(videoFrames);
        }
        else
        {
            return PredictOnnx(videoFrames);
        }
    }

    /// <summary>
    /// Gets the top-K predicted action classes with probabilities.
    /// </summary>
    /// <param name="videoFrames">Video frames tensor.</param>
    /// <param name="topK">Number of top predictions to return.</param>
    /// <returns>List of (classIndex, probability) pairs sorted by probability.</returns>
    public List<(int ClassIndex, double Probability)> GetTopKPredictions(Tensor<T> videoFrames, int topK = 5)
    {
        var probabilities = Classify(videoFrames);

        var results = new List<(int, double)>();
        for (int i = 0; i < probabilities.Length; i++)
        {
            results.Add((i, Convert.ToDouble(probabilities.Data.Span[i])));
        }

        return results.OrderByDescending(x => x.Item2).Take(topK).ToList();
    }

    /// <summary>
    /// Extracts video features before the classification head.
    /// </summary>
    /// <param name="videoFrames">Video frames tensor.</param>
    /// <returns>Feature embedding tensor.</returns>
    public Tensor<T> ExtractFeatures(Tensor<T> videoFrames)
    {
        if (videoFrames is null)
            throw new ArgumentNullException(nameof(videoFrames));

        var result = PrepareNativeInput(videoFrames, out var frameCount);
        for (int i = 0; i < Layers.Count - 1; i++)
        {
            result = ForwardLayer(result, Layers[i], frameCount);
        }
        return result;
    }

    #endregion

    #region Inference

    private Tensor<T> Forward(Tensor<T> input)
    {
        var logits = ForwardNativeLogits(input);
        return Softmax(logits);
    }

    private Tensor<T> ForwardNativeLogits(Tensor<T> input)
    {
        var result = PrepareNativeInput(input, out var frameCount);
        foreach (var layer in Layers)
        {
            result = ForwardLayer(result, layer, frameCount);
        }
        return result;
    }

    private Tensor<T> ForwardLayer(Tensor<T> input, ILayer<T> layer, int frameCount)
        => layer is TimeSformerBlockLayer<T> timeSformerBlock
            ? timeSformerBlock.Forward(input, frameCount)
            : layer.Forward(input);

    private Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            inputData[i] = Convert.ToSingle(input.Data.Span[i]);
        }

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
        var inputMeta = _onnxSession.InputMetadata;
        string inputName = inputMeta.Keys.First();

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, onnxInput)
        };

        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputShape = outputTensor.Dimensions.ToArray();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
        {
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    private Tensor<T> Softmax(Tensor<T> logits)
    {
        var result = new Tensor<T>(logits._shape);
        double maxVal = double.MinValue;

        for (int i = 0; i < logits.Length; i++)
        {
            double val = Convert.ToDouble(logits.Data.Span[i]);
            if (val > maxVal) maxVal = val;
        }

        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            sum += Math.Exp(Convert.ToDouble(logits.Data.Span[i]) - maxVal);
        }

        for (int i = 0; i < logits.Length; i++)
        {
            double prob = Math.Exp(Convert.ToDouble(logits.Data.Span[i]) - maxVal) / sum;
            result.Data.Span[i] = NumOps.FromDouble(prob);
        }

        return result;
    }

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        return Classify(input);
    }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode.");

        EnsureLayerRandomSeedsWired();
        return ForwardNativeLogits(input);
    }

    /// <inheritdoc/>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (!_useNativeMode || !_usesTokenizedNativePath)
            return base.GetNamedLayerActivations(input);

        var activations = new Dictionary<string, Tensor<T>>();
        var current = PrepareNativeInput(input, out var frameCount);

        for (int i = 0; i < Layers.Count; i++)
        {
            current = ForwardLayer(current, Layers[i], frameCount);
            activations[$"Layer_{i}_{Layers[i].GetType().Name}"] = current.Clone();
        }

        return activations;
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    #endregion

    #region Layer Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
        {
            ClearLayers();
            return;
        }

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _usesTokenizedNativePath = false;
        }
        else
        {
            int inputChannels = Architecture.InputDepth > 0 ? Architecture.InputDepth : 3;
            int inputHeight = Architecture.InputHeight > 0 ? Architecture.InputHeight : 224;
            int inputWidth = Architecture.InputWidth > 0 ? Architecture.InputWidth : 224;

            Layers.AddRange(LayerHelper<T>.CreateDefaultTimeSformerLayers(
                inputChannels: inputChannels,
                inputHeight: inputHeight,
                inputWidth: inputWidth,
                embedDim: _embedDim,
                numHeads: _numHeads,
                numLayers: _numLayers,
                numFrames: _numFrames,
                patchSize: _patchSize,
                numClasses: _numClasses,
                useJointSpaceTimeAttention: _attentionType == AttentionType.JointSpaceTime));
            _usesTokenizedNativePath = true;
        }
    }

    private Tensor<T> PrepareNativeInput(Tensor<T> input, out int frameCount)
    {
        if (!_usesTokenizedNativePath)
        {
            frameCount = _numFrames;
            return input;
        }

        return TokenizeVideo(input, out frameCount);
    }

    private Tensor<T> TokenizeVideo(Tensor<T> input, out int frameCount)
    {
        if (input is null)
            throw new ArgumentNullException(nameof(input));

        int batch;
        int frames;
        int channels;
        int height;
        int width;
        bool hasBatch;

        switch (input.Rank)
        {
            case 5:
                batch = input.Shape[0];
                frames = input.Shape[1];
                channels = input.Shape[2];
                height = input.Shape[3];
                width = input.Shape[4];
                hasBatch = true;
                break;
            case 4:
                batch = 1;
                frames = input.Shape[0];
                channels = input.Shape[1];
                height = input.Shape[2];
                width = input.Shape[3];
                hasBatch = false;
                break;
            case 3:
                batch = 1;
                frames = 1;
                channels = input.Shape[0];
                height = input.Shape[1];
                width = input.Shape[2];
                hasBatch = false;
                break;
            default:
                throw new ArgumentException(
                    "TimeSformer native mode expects [B,T,C,H,W], [T,C,H,W], or [C,H,W] input.",
                    nameof(input));
        }

        if (batch <= 0 || frames <= 0 || channels <= 0 || height <= 0 || width <= 0)
            throw new ArgumentException("TimeSformer input dimensions must all be positive.", nameof(input));

        int patch = Math.Min(_patchSize, Math.Min(height, width));
        patch = Math.Max(1, patch);
        int patchRows = Math.Max(1, height / patch);
        int patchCols = Math.Max(1, width / patch);
        int spatialPatches = patchRows * patchCols;
        int patchDim = channels * patch * patch;
        int tokensPerSample = frames * spatialPatches;
        var tokens = new Tensor<T>([batch, tokensPerSample, patchDim]);

        var source = input.Data.Span;
        var destination = tokens.Data.Span;

        for (int b = 0; b < batch; b++)
        {
            for (int f = 0; f < frames; f++)
            {
                for (int py = 0; py < patchRows; py++)
                {
                    for (int px = 0; px < patchCols; px++)
                    {
                        int tokenIndex = f * spatialPatches + py * patchCols + px;
                        int feature = 0;
                        for (int c = 0; c < channels; c++)
                        {
                            for (int y = 0; y < patch; y++)
                            {
                                int sourceY = py * patch + y;
                                for (int x = 0; x < patch; x++)
                                {
                                    int sourceX = px * patch + x;
                                    int sourceIndex = hasBatch
                                        ? (((b * frames + f) * channels + c) * height + sourceY) * width + sourceX
                                        : input.Rank == 4
                                            ? (((f * channels + c) * height + sourceY) * width + sourceX)
                                            : ((c * height + sourceY) * width + sourceX);
                                    int destIndex = (b * tokensPerSample + tokenIndex) * patchDim + feature++;
                                    destination[destIndex] = source[sourceIndex];
                                }
                            }
                        }
                    }
                }
            }
        }

        frameCount = frames;
        return tokens;
    }

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Parameter updates are not supported in ONNX mode.");

        int offset = 0;
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            int paramCount = layerParams.Length;
            if (paramCount > 0 && offset + paramCount <= parameters.Length)
            {
                var slice = new Vector<T>(paramCount);
                for (int i = 0; i < paramCount; i++)
                {
                    slice[i] = parameters[offset + i];
                }
                layer.SetParameters(slice);
                offset += paramCount;
            }
        }
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var additionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "TimeSformer" },
            { "EmbedDim", _embedDim },
            { "NumHeads", _numHeads },
            { "NumLayers", _numLayers },
            { "NumFrames", _numFrames },
            { "PatchSize", _patchSize },
            { "ImageSize", _imageSize },
            { "NumClasses", _numClasses },
            { "AttentionType", _attentionType.ToString() },
            { "UseNativeMode", _useNativeMode }
        };

        if (!_useNativeMode && _onnxModelPath != null)
        {
            additionalInfo["OnnxModelPath"] = _onnxModelPath;
        }

        return new ModelMetadata<T>
        {
            AdditionalInfo = additionalInfo,
            ModelData = _useNativeMode ? this.Serialize() : []
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Serialization is not supported in ONNX mode.");

        writer.Write(_embedDim);
        writer.Write(_numHeads);
        writer.Write(_numLayers);
        writer.Write(_numFrames);
        writer.Write(_patchSize);
        writer.Write(_imageSize);
        writer.Write(_numClasses);
        writer.Write((int)_attentionType);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Deserialization is not supported in ONNX mode.");

        _ = reader.ReadInt32(); // embedDim
        _ = reader.ReadInt32(); // numHeads
        _ = reader.ReadInt32(); // numLayers
        _ = reader.ReadInt32(); // numFrames
        _ = reader.ReadInt32(); // patchSize
        _ = reader.ReadInt32(); // imageSize
        _ = reader.ReadInt32(); // numClasses
        _ = reader.ReadInt32(); // attentionType
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new TimeSformer<T>(
            Architecture,
            numClasses: _numClasses,
            optimizer: null,
            lossFunction: _lossFunction,
            embedDim: _embedDim,
            numHeads: _numHeads,
            numLayers: _numLayers,
            numFrames: _numFrames,
            patchSize: _patchSize,
            attentionType: _attentionType,
            options: _options);
    }

    #endregion
}

/// <summary>
/// Attention type for TimeSformer.
/// </summary>
public enum AttentionType
{
    /// <summary>
    /// Joint space-time attention (compute attention over all patches from all frames).
    /// </summary>
    JointSpaceTime,

    /// <summary>
    /// Divided space-time attention (separate temporal and spatial attention).
    /// </summary>
    DividedSpaceTime,

    /// <summary>
    /// Sparse local-global attention.
    /// </summary>
    SparseLG
}
