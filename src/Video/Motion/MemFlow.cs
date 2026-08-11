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

namespace AiDotNet.Video.Motion;

/// <summary>
/// MemFlow optical flow with memory for real-time historical motion aggregation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "MemFlow: Optical Flow Estimation and Prediction with Memory" (Dong et al., CVPR 2024)</item>
/// </list></para>
/// <para><b>For Beginners:</b> MemFlow uses memory-efficient transformers for optical flow estimation. It reduces memory consumption while maintaining high accuracy through a chunked attention mechanism.</para>
/// <para>
/// MemFlow augments flow estimation with an explicit memory module that aggregates historical motion information for improved temporal consistency.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a MemFlow model for memory-augmented optical flow estimation
/// var memFlow = new MemFlow&lt;double&gt;();
///
/// // Or configure with custom parameters
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3, outputSize: 2);
/// var model = new MemFlow&lt;double&gt;(architecture);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("MemFlow: Optical Flow Estimation and Prediction with Memory",
    "https://arxiv.org/abs/2404.04808",
    Year = 2024,
    Authors = "Qiaole Dong, Yanwei Fu")]
public partial class MemFlow<T> : OpticalFlowBase<T>
{
    private readonly MemFlowOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _numFeatures;
    private int _numLayers;
    private ConvolutionalLayer<T>? _featureExtract;
    private readonly List<ConvolutionalLayer<T>> _processingBlocks;
    private ConvolutionalLayer<T>? _outputConv;

    /// <summary>
    /// Creates a new MemFlow model for native training and inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFeatures">Number of feature channels. Default: 64.</param>
    /// <param name="numLayers">Number of processing layers. Default: 8.</param>
    /// <param name="options">Optional configuration options.</param>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public MemFlow()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.ThreeDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            // 2 frames stacked channel-wise (2×3=6): the lazy _featureExtract conv is sized from
            // InputDepth by ResolveLazyLayerShapes, and EstimateFlow feeds it the concatenated pair,
            // so it must be 6 not 3. Single-encoder flow models only (RAFT/GMFlow have a separate
            // 3-channel context encoder and are excluded). PredictCore splits per-frame via Shape[1]/2.
            inputHeight: 256, inputWidth: 256, inputDepth: 6,
            outputSize: 2))
    {
    }

    public MemFlow(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 64,
        int numLayers = 8,
        MemFlowOptions? options = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        if (numFeatures <= 0)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), numFeatures, "Number of features must be positive.");
        if (numLayers <= 0)
            throw new ArgumentOutOfRangeException(nameof(numLayers), numLayers, "Number of layers must be positive.");
        _options = options ?? new MemFlowOptions();
        Options = _options;

        _numFeatures = numFeatures;
        _numLayers = numLayers;
        _processingBlocks = [];

        InitializeNativeLayers(architecture);
    }

    private void InitializeNativeLayers(NeuralNetworkArchitecture<T> arch)
    {
        int height = arch.InputHeight > 0 ? arch.InputHeight : 64;
        int width = arch.InputWidth > 0 ? arch.InputWidth : 64;
        int channels = arch.InputDepth > 0 ? arch.InputDepth : 3;

        _featureExtract = new ConvolutionalLayer<T>(_numFeatures, 3, 1, 1);

        for (int i = 0; i < _numLayers; i++)
        {
            _processingBlocks.Add(new ConvolutionalLayer<T>(_numFeatures, 3, 1, 1));
        }

        _outputConv = new ConvolutionalLayer<T>(2, 3, 1, 1);

        InitializeLayers();
    }

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        ClearLayers();

        if (_featureExtract is not null)
            Layers.Add(_featureExtract);
        foreach (var block in _processingBlocks)
            Layers.Add(block);
        if (_outputConv is not null)
            Layers.Add(_outputConv);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames)
    {
        return NormalizeFrames(rawFrames);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        return DenormalizeFrames(modelOutput);
    }

    /// <inheritdoc/>
    public override Tensor<T> EstimateFlow(Tensor<T> frame0, Tensor<T> frame1)
    {
        if (frame0.Rank < 3)
            throw new ArgumentException($"frame0 must be at least rank 3 [C,H,W], got rank {frame0.Rank}.", nameof(frame0));
        if (frame1.Rank < 3)
            throw new ArgumentException($"frame1 must be at least rank 3 [C,H,W], got rank {frame1.Rank}.", nameof(frame1));
        if (frame0.Shape[0] != frame1.Shape[0] || frame0.Shape[1] != frame1.Shape[1] || frame0.Shape[2] != frame1.Shape[2])
            throw new ArgumentException(
                $"Frame shapes must match. frame0: [{string.Join(",", frame0._shape)}], frame1: [{string.Join(",", frame1._shape)}].",
                nameof(frame1));
        int height = frame0.Shape[1];
        int width = frame0.Shape[2];

        // Concatenate frames as input pair
        var concat = ConcatenateFeatures(frame0, frame1);
        if (_featureExtract is null || _outputConv is null)
            throw new InvalidOperationException("Model layers not initialized.");

        var feat = _featureExtract.Forward(concat);
        foreach (var block in _processingBlocks)
        {
            feat = block.Forward(feat);
        }
        var rawFlow = _outputConv.Forward(feat);

        // The output convolution already emits exactly 2 channels at the input resolution
        // (ConvolutionalLayer(2, kernel 3, stride 1, padding 1)), so rawFlow IS the flow field. The
        // element-by-element Data.Span copy this replaced was a numeric no-op that severed the
        // autodiff tape at the end of the forward pass, discarding the gradient path for the whole
        // network behind it. Returning the tensor directly is bit-identical; the guard is kept so a
        // layer misconfiguration still fails loudly instead of silently yielding a wrong-shaped field.
        int expectedLength = 2 * height * width;
        if (rawFlow.Length < expectedLength)
        {
            throw new InvalidOperationException(
                $"Raw flow output ({rawFlow.Length} elements) is smaller than the expected flow field " +
                $"({expectedLength} elements for 2x{height}x{width}).");
        }

        return rawFlow;
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
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

    // UpdateParameters restated the base verbatim; ModelBase routes it to SetParameters.
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "MemFlow" },
                { "NumFeatures", _numFeatures },
                { "NumLayers", _numLayers }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_numFeatures);
        writer.Write(_numLayers);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _numFeatures = reader.ReadInt32();
        _numLayers = reader.ReadInt32();

        // Re-link the cached role fields to the layers the BASE already deserialized (with their
        // trained, shape-resolved weights) — do NOT call InitializeNativeLayers, which allocates
        // FRESH random-initialized convolutions and, via InitializeLayers, replaces the deserialized
        // layers in Layers. Doing so discarded the trained weights so a cloned/loaded model predicted
        // from random init (#1221 class: Clone_AfterTraining / Clone_ShouldProduceIdenticalOutput).
        // Layer order matches InitializeLayers: [featureExtract, ...processingBlocks, outputConv].
        if (Layers.Count < _numLayers + 2)
            throw new InvalidDataException(
                $"MemFlow serialized layer count {Layers.Count} is too small for {_numLayers} processing blocks.");

        _featureExtract = Layers[0] as ConvolutionalLayer<T>
            ?? throw new InvalidDataException("MemFlow feature extractor layer is missing or has the wrong type.");

        _processingBlocks.Clear();
        for (int i = 0; i < _numLayers; i++)
        {
            _processingBlocks.Add(Layers[i + 1] as ConvolutionalLayer<T>
                ?? throw new InvalidDataException($"MemFlow processing block {i} is missing or has the wrong type."));
        }

        _outputConv = Layers[_numLayers + 1] as ConvolutionalLayer<T>
            ?? throw new InvalidDataException("MemFlow output layer is missing or has the wrong type.");
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new MemFlow<T>(Architecture, _numFeatures, _numLayers, _options);
    }
}
