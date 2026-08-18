using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.Diffusion;

/// <summary>
/// ODISE Segmentation: Panoptic segmentation via diffusion model features.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Diffusion-based panoptic segmentation. Open-vocabulary scene understanding.
///
/// Common use cases:
/// - Diffusion-based panoptic segmentation
/// - Open-vocabulary scene understanding
/// - Text-guided segmentation via diffusion
/// - Novel category segmentation
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Stable Diffusion UNet features for dense prediction
/// - CLIP text encoder for open-vocabulary category matching
/// - Mask generator trained on frozen diffusion features
/// - Joint text-image representation for panoptic labels
/// </para>
/// <para>
/// <b>Reference:</b> Xu et al., "Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models", CVPR 2023.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an ODISE model for open-vocabulary panoptic segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 512, inputWidth: 512, inputDepth: 3, outputSize: 150);
/// var model = new ODISESegmentation&lt;double&gt;(architecture, numClasses: 150);
///
/// // Or load a pre-trained ONNX model
/// var onnxModel = new ODISESegmentation&lt;double&gt;(architecture, "odise_model.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models", "https://arxiv.org/abs/2303.04803", Year = 2023, Authors = "Xu et al.")]
public partial class ODISESegmentation<T> : Common.PanopticSegmentationBase<T>
{
    private readonly ODISESegmentationOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    // Only ODISE's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from PanopticSegmentationBase -> SegmentationModelBase.
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    #endregion

    #region Properties
    /// <summary>
    /// Gets whether using native mode (trainable) or ONNX mode (inference only).
    /// </summary>
    internal bool UseNativeMode => _useNativeMode;
    #endregion

    /// <summary>
    /// ODISE's stuff/thing split: the first third of the class list is "stuff" (amorphous regions
    /// like sky or road), the rest are countable "things". Kept as a static helper so both
    /// constructors can hand the same split to the panoptic base before any field is assigned.
    /// </summary>
    private static int StuffClassCount(int numClasses) => Math.Max(1, numClasses / 3);

    #region Constructors
    /// <summary>
    /// Initializes ODISESegmentation in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 133).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable ODISESegmentation model.
    /// </para>
    /// </remarks>
    public ODISESegmentation(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 133,
        double dropRate = 0.1,
        ODISESegmentationOptions? options = null)
        : base(architecture, optimizer, lossFunction, numClasses,
               StuffClassCount(numClasses), numClasses - StuffClassCount(numClasses))
    {
        _options = options ?? new ODISESegmentationOptions(); Options = _options;
        _dropRate = dropRate;
        ValidateArchitectureOptions(_options);
        _channelDims = (int[])_options.ChannelDimensions.Clone();
        _depths = (int[])_options.StageDepths.Clone();
        _decoderDim = _options.DecoderDimension;
        InitializeLayers();
    }

    /// <summary>
    /// Initializes ODISESegmentation in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 133).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained ODISESegmentation from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public ODISESegmentation(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 133,
        ODISESegmentationOptions? options = null)
        : base(architecture, onnxModelPath, numClasses,
               StuffClassCount(numClasses), numClasses - StuffClassCount(numClasses))
    {
        _options = options ?? new ODISESegmentationOptions(); Options = _options;
        _dropRate = 0.1;
        ValidateArchitectureOptions(_options);
        _channelDims = (int[])_options.ChannelDimensions.Clone();
        _depths = (int[])_options.StageDepths.Clone();
        _decoderDim = _options.DecoderDimension;
        InitializeLayers();
    }
    #endregion

    #region Public Methods
    /// <summary>
    /// Performs one training step.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="expectedOutput">Ground-truth segmentation tensor.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Trains the model. Only available in native mode.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when called on an ONNX-mode model.</exception>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode) throw new InvalidOperationException("Training is not supported in ONNX mode. Use the native mode constructor for training.");
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
    private static void ValidateArchitectureOptions(ODISESegmentationOptions options)
    {
        if (options.ChannelDimensions is null)
            throw new ArgumentNullException(nameof(options.ChannelDimensions));
        if (options.StageDepths is null)
            throw new ArgumentNullException(nameof(options.StageDepths));
        if (options.ChannelDimensions.Length != 4 || options.StageDepths.Length != 4)
        {
            throw new ArgumentException(
                "ODISE ChannelDimensions and StageDepths must each contain four stages.",
                nameof(options));
        }
        if (options.ChannelDimensions.Any(value => value <= 0) ||
            options.StageDepths.Any(value => value <= 0) ||
            options.DecoderDimension <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                "All ODISE channel dimensions, stage depths, and the decoder dimension must be positive.");
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++) features = Layers[i].Forward(features);
        for (int i = _encoderLayerEnd; i < Layers.Count; i++) features = Layers[i].Forward(features);
        if (!hasBatch) features = RemoveBatchDimension(features); return features;
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null) throw new InvalidOperationException("ONNX session is not initialized.");
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++) inputData[i] = Convert.ToSingle(input.Data.Span[i]);
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "images";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };
        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++) outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        var result = new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
        if (!hasBatch) result = RemoveBatchDimension(result); return result;
    }

    // AddBatchDimension / RemoveBatchDimension are inherited from SegmentationModelBase.
    #endregion

    #region Abstract Implementation
    /// <summary>
    /// Initializes the encoder and decoder layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In native mode, builds the neural network layers.
    /// In ONNX mode, no layers are created.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) { ClearLayers(); return; }
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        { Layers.AddRange(Architecture.Layers); _encoderLayerEnd = Architecture.Layers.Count / 2; }
        else
        {
            var encoderLayers = LayerHelper<T>.CreateODISESegmentationEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateODISESegmentationDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
            Layers.AddRange(decoderLayers);
        }
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <summary>
    /// Collects metadata describing this model's configuration.
    /// </summary>
    /// <returns>Model metadata.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns a summary for saving or display.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "ODISESegmentation" }, { "SegmentationType", "Panoptic" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
        ModelData = SerializeForMetadata()
    };

    /// <summary>
    /// Writes configuration to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Saves model configuration for later reconstruction.
    /// </para>
    /// </remarks>


    /// <summary>
    /// Reads configuration from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads model configuration when restoring a saved model.
    /// </para>
    /// </remarks>


    /// <summary>
    /// Creates a new instance with the same configuration but fresh weights.
    /// </summary>
    /// <returns>A new model instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a copy for cross-validation or ensemble training.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new ODISESegmentationOptions(_options);
        return _useNativeMode
            ? new ODISESegmentation<T>(Architecture, optimizer: null, lossFunction: LossFunction,
                numClasses: _numClasses, dropRate: _dropRate, options: options)
            : new ODISESegmentation<T>(Architecture,
                _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."),
                _numClasses, options);
    }

    // Dispose of the ONNX session and the _disposed latch are handled by SegmentationModelBase.
    #endregion

    #region IPanopticSegmentation Implementation
    // NumClasses / InputHeight / InputWidth / IsOnnxMode / Segment / NumStuffClasses /
    // NumThingClasses all arrive from PanopticSegmentationBase.

    /// <inheritdoc />
    public override PanopticSegmentationResult<T> SegmentPanoptic(Tensor<T> image)
    {
        var logits = Common.SegmentationTensorOps.EnsureUnbatched(Predict(image));
        var probMap = Common.SegmentationTensorOps.SoftmaxAlongClassDim(logits);
        var semanticMap = Common.SegmentationTensorOps.ArgmaxAlongClassDim(logits);
        int h = semanticMap.Shape[0], w = semanticMap.Shape[1];
        int numStuff = NumStuffClasses;
        var instanceMap = new Tensor<T>([h, w]);
        var panopticMap = new Tensor<T>([h, w]);
        var segments = new List<PanopticSegment<T>>();
        int nextInstId = 1;
        for (int cls = 0; cls < numStuff; cls++)
        {
            int area = 0; double sumConf = 0;
            for (int row = 0; row < h; row++)
                for (int col = 0; col < w; col++)
                    if (NumOps.Compare(semanticMap[row, col], NumOps.FromDouble(cls)) == 0)
                    { panopticMap[row, col] = NumOps.FromDouble(cls * 1000); area++; sumConf += NumOps.ToDouble(probMap[cls, row, col]); }
            if (area > 0) segments.Add(new PanopticSegment<T> { SegmentId = cls, ClassId = cls, IsThing = false, Confidence = sumConf / area, Area = area });
        }
        for (int cls = numStuff; cls < _numClasses; cls++)
        {
            var (labelMap, count) = Common.SegmentationTensorOps.LabelConnectedComponents(semanticMap, cls);
            for (int comp = 1; comp <= count; comp++)
            {
                int instId = nextInstId++;
                int area = 0; double sumConf = 0; var compMask = new Tensor<T>([h, w]);
                for (int row = 0; row < h; row++)
                    for (int col = 0; col < w; col++)
                        if (NumOps.Compare(labelMap[row, col], NumOps.FromDouble(comp)) == 0)
                        { instanceMap[row, col] = NumOps.FromDouble(instId); panopticMap[row, col] = NumOps.FromDouble(cls * 1000 + instId); compMask[row, col] = NumOps.FromDouble(1.0); area++; sumConf += NumOps.ToDouble(probMap[cls, row, col]); }
                if (area > 0) segments.Add(new PanopticSegment<T> { SegmentId = instId, ClassId = cls, IsThing = true, Confidence = sumConf / area, Area = area, Mask = compMask });
            }
        }
        return new PanopticSegmentationResult<T> { SemanticMap = semanticMap, InstanceMap = instanceMap, PanopticMap = panopticMap, Segments = segments };
    }
    #endregion
}
