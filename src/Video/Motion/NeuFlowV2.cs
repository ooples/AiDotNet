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
/// NeuFlow v2 high-efficiency optical flow on edge devices.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "NeuFlow v2: Push High-Efficiency Optical Flow To the Limit" (Zhang et al., 2024)</item>
/// </list></para>
/// <para><b>For Beginners:</b> NeuFlow V2 is a fast, lightweight optical flow estimator designed for real-time applications. It achieves good accuracy with significantly reduced computation compared to transformer-based methods.</para>
/// <para>
/// NeuFlow v2 achieves high-efficiency optical flow estimation suitable for edge devices through a lightweight backbone and optimized inference.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a NeuFlow V2 model for efficient edge-device optical flow
/// var neuFlow = new NeuFlowV2&lt;double&gt;();
///
/// // Or configure with custom parameters
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3, outputSize: 2);
/// var model = new NeuFlowV2&lt;double&gt;(architecture);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
// Title corrected to the published form; the arXiv id was already right.
[ResearchPaper("NeuFlow v2: Push High-Efficiency Optical Flow To the Limit",
    "https://arxiv.org/abs/2408.10161",
    Year = 2024,
    Authors = "Zhiyong Zhang, Anurag Ranjan, Huaizu Jiang")]
public partial class NeuFlowV2<T> : OpticalFlowBase<T>
{
    private readonly NeuFlowV2Options _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _numFeatures;
    private int _numLayers;
    private ConvolutionalLayer<T>? _featureExtract;
    private readonly List<ConvolutionalLayer<T>> _processingBlocks;
    private ConvolutionalLayer<T>? _outputConv;

    /// <summary>
    /// Creates a new NeuFlowV2 model for native training and inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFeatures">Number of feature channels. Default: 64.</param>
    /// <param name="numLayers">Number of processing layers. Default: 8.</param>
    /// <param name="options">Optional configuration options.</param>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public NeuFlowV2()
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

    public NeuFlowV2(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 64,
        int numLayers = 8,
        NeuFlowV2Options? options = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        if (numFeatures <= 0)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), numFeatures, "Number of features must be positive.");
        if (numLayers <= 0)
            throw new ArgumentOutOfRangeException(nameof(numLayers), numLayers, "Number of layers must be positive.");
        _options = options ?? new NeuFlowV2Options();
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
                { "ModelName", "NeuFlowV2" },
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
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new NeuFlowV2<T>(Architecture, _numFeatures, _numLayers, _options);
    }
}
