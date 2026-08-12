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
/// RPKNet recurrent partial kernel network with separable large kernels for flow.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "RPKNet: Recurrent Partial Kernel Network for Efficient Optical Flow" (Morimitsu et al., AAAI 2024)</item>
/// </list></para>
/// <para><b>For Beginners:</b> RPKNet (Recurrent Position-aware Kernel Network) uses position-aware convolution kernels that adapt to each pixel position for accurate optical flow estimation.</para>
/// <para>
/// RPKNet uses recurrent partial kernel processing with separable large kernels for variable multi-scale feature extraction in optical flow.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an RPKNet model for partial kernel optical flow estimation
/// var rpkNet = new RPKNet&lt;double&gt;();
///
/// // Or configure with custom parameters
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3, outputSize: 2);
/// var model = new RPKNet&lt;double&gt;(architecture);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
// No arXiv preprint exists for RPKNet; AAAI proceedings is the canonical venue.
[ResearchPaper("Recurrent Partial Kernel Network for Efficient Optical Flow Estimation",
    "https://ojs.aaai.org/index.php/AAAI/article/view/28224",
    Year = 2024,
    Authors = "Henrique Morimitsu, Xiaobin Zhu, Xiangyang Ji, Xu-Cheng Yin")]
public partial class RPKNet<T> : OpticalFlowBase<T>
{
    private readonly RPKNetOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _numFeatures;
    private int _numLayers;
    private ConvolutionalLayer<T>? _featureExtract;
    private readonly List<ConvolutionalLayer<T>> _processingBlocks;
    private ConvolutionalLayer<T>? _outputConv;

    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public RPKNet()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.ThreeDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            // Optical flow consumes the two RGB frames stacked channel-wise (2×3=6). The lazily
            // resolved feature-extractor conv is sized by ResolveLazyLayerShapes from this
            // InputDepth, and EstimateFlow feeds it the concatenated 6-channel pair — so InputDepth
            // must be 6, not the single-frame 3 (which resolved the conv to 3 and made the real
            // forward throw "Expected input depth 3, but got 6"). PredictCore still splits the input
            // into two 3-channel frames via input.Shape[1]/2.
            inputHeight: 256, inputWidth: 256, inputDepth: 6,
            outputSize: 2))
    {
    }

    /// <summary>
    /// Creates a new RPKNet model for native training and inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFeatures">Number of feature channels. Default: 64.</param>
    /// <param name="numLayers">Number of processing layers. Default: 8.</param>
    /// <param name="options">Optional configuration options.</param>
    public RPKNet(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 64,
        int numLayers = 8,
        RPKNetOptions? options = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        if (numFeatures <= 0)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), numFeatures, "Number of features must be positive.");
        if (numLayers <= 0)
            throw new ArgumentOutOfRangeException(nameof(numLayers), numLayers, "Number of layers must be positive.");
        _options = options ?? new RPKNetOptions();
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
                { "ModelName", "RPKNet" },
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
}
