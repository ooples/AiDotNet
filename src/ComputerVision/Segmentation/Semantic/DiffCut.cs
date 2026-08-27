using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.Semantic;

/// <summary>
/// DiffCut: Catalyzing Zero-Shot Semantic Segmentation with Diffusion Features and Recursive Normalized Cut.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DiffCut is a unique semantic segmentation model that requires no training
/// labels at all. It extracts features from a diffusion model's internal UNet representations and
/// applies a graph-based algorithm called Normalized Cut to partition the image into semantically
/// meaningful regions. This "zero-shot" approach means you can segment images without ever training
/// on segmentation labels.
///
/// Common use cases:
/// - Zero-shot segmentation when no labeled data is available
/// - Exploring and annotating new datasets
/// - Domain adaptation where labeled data doesn't exist
/// - Research into unsupervised visual understanding
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Extracts intermediate features from a pre-trained Stable Diffusion UNet
/// - Builds an affinity graph from diffusion feature similarities
/// - Applies recursive Normalized Cut (NCut) for hierarchical segmentation
/// - Achieves +7.3 mIoU over prior SOTA on unsupervised segmentation benchmarks
/// - Training-free: no fine-tuning required
/// </para>
/// <para>
/// <b>Reference:</b> Couairon et al., "DiffCut: Catalyzing Zero-Shot Semantic Segmentation
/// with Diffusion Features and Recursive Normalized Cut", NeurIPS 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a DiffCut model for zero-shot semantic segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 512, inputWidth: 512, inputDepth: 3, outputSize: 150);
/// var model = new DiffCut&lt;double&gt;(architecture, numClasses: 150);
///
/// // Or load a pre-trained ONNX model for unsupervised segmentation
/// var onnxModel = new DiffCut&lt;double&gt;(architecture, "diffcut.onnx", numClasses: 150);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("DiffCut: Catalyzing Zero-Shot Semantic Segmentation with Diffusion Features and Recursive Normalized Cut", "https://arxiv.org/abs/2406.02842", Year = 2024, Authors = "Couairon et al.")]
public partial class DiffCut<T> : Common.SemanticSegmentationBase<T>
{
    private readonly DiffCutOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields

    // Only DiffCut's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from SemanticSegmentationBase -> SegmentationModelBase.
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;

    #endregion

    #region Properties

    // SupportsTraining and NumClasses are inherited from SegmentationModelBase and say exactly the
    // same thing, so re-declaring them here would only create two sources of one fact.
    internal bool UseNativeMode => _useNativeMode;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes DiffCut in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW, consistent with diffusion
    /// model fine-tuning practices).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of semantic classes (default: 150).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable DiffCut model. While DiffCut is designed for
    /// zero-shot segmentation (no training needed), this mode allows optional fine-tuning of
    /// the diffusion feature extractor for specific domains.
    /// </para>
    /// </remarks>
    public DiffCut(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numClasses = 150,
        double dropRate = 0.1,
        DiffCutOptions? options = null)
        : base(architecture, optimizer, lossFunction, numClasses)
    {
        _options = options ?? new DiffCutOptions();
        Options = _options;
        _dropRate = dropRate;

        _channelDims = [64, 128, 256, 512];
        _depths = [2, 2, 4, 2];
        _decoderDim = 256;

        InitializeLayers();
    }

    /// <summary>
    /// Initializes DiffCut in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of classes (default: 150).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained DiffCut for fast zero-shot inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if file not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if ONNX load fails.</exception>
    public DiffCut(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 150,
        DiffCutOptions? options = null)
        : base(architecture, onnxModelPath, numClasses)
    {
        _options = options ?? new DiffCutOptions();
        Options = _options;
        _dropRate = 0.0;

        _channelDims = [64, 128, 256, 512];
        _depths = [2, 2, 4, 2];
        _decoderDim = 256;

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Runs a forward pass to produce per-pixel segmentation logits.
    /// </summary>
    /// <param name="input">Input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Per-pixel class logits tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Extracts diffusion features and applies Normalized Cut to segment
    /// the image into meaningful regions, then classifies each region.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        return !_useNativeMode ? PredictOnnx(input) : Forward(input);
    }

    /// <summary>
    /// Performs one training step.
    /// </summary>
    /// <param name="input">Input image tensor.</param>
    /// <param name="expectedOutput">Ground-truth segmentation map.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Optional fine-tuning of the diffusion feature extractor.
    /// DiffCut is designed for zero-shot use, but training can improve domain-specific performance.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown in ONNX mode.</exception>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode.");

        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput, Optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    #endregion

    #region Private Methods

    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);
        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++) features = Layers[i].Forward(features);
        for (int i = _encoderLayerEnd; i < Layers.Count; i++) features = Layers[i].Forward(features);
        if (!hasBatch) features = RemoveBatchDimension(features);
        return features;
    }

    protected override Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null) throw new InvalidOperationException("ONNX session is not initialized.");
        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);
        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++) inputData[i] = Convert.ToSingle(input.Data.Span[i]);
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "pixel_values";
        using var results = _onnxSession.Run(new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) });
        var outputTensor = results.First().AsTensor<float>();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++) outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        var result = new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
        if (!hasBatch) result = RemoveBatchDimension(result);
        return result;
    }

    #endregion

    #region Abstract Implementation

    /// <summary>
    /// Initializes the diffusion UNet encoder and Normalized Cut decoder layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In native mode, creates layers that mimic the diffusion model's UNet
    /// feature extraction (encoder) and Normalized Cut classification (decoder). In ONNX mode,
    /// no layers are created.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) { ClearLayers(); return; }
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = Architecture.Layers.Count / 2;
        }
        else
        {
            var encoderLayers = LayerHelper<T>.CreateDiffCutEncoderLayers(
                _channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count;
            Layers.AddRange(encoderLayers);

            int[] patchKernels = [7, 3, 3, 3]; int[] patchStrides = [4, 2, 2, 2]; int[] patchPaddings = [3, 1, 1, 1];
            int featureH = _height, featureW = _width;
            for (int stage = 0; stage < 4; stage++)
            {
                featureH = (featureH + 2 * patchPaddings[stage] - patchKernels[stage]) / patchStrides[stage] + 1;
                featureW = (featureW + 2 * patchPaddings[stage] - patchKernels[stage]) / patchStrides[stage] + 1;
            }

            Layers.AddRange(LayerHelper<T>.CreateDiffCutDecoderLayers(
                _channelDims[^1], _decoderDim, _numClasses, featureH, featureW));
        }
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <summary>
    /// Collects model metadata.
    /// </summary>
    /// <returns>Model metadata.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Summary of the model for saving, comparing, or display.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "DiffCut" }, { "Description", "DiffCut Zero-Shot Semantic Segmentation" },
                { "InputHeight", _height }, { "InputWidth", _width }, { "InputChannels", _channels },
                { "NumClasses", _numClasses }, { "DecoderDim", _decoderDim }, { "DropRate", _dropRate },
                { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count }
            },
            ModelData = SerializeForMetadata()
        };
    }

    #endregion
}
