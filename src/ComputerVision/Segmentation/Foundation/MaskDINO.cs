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

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Mask DINO extends the powerful DINO object detector with a mask prediction
/// branch, creating a unified architecture that handles object detection, instance segmentation,
/// panoptic segmentation, and semantic segmentation all in one model. Instead of building separate
/// models for each task, Mask DINO uses a shared backbone and query-based transformer to do everything.
///
/// Common use cases:
/// - Joint object detection + instance segmentation
/// - Panoptic segmentation (detecting all things and stuff)
/// - Research requiring a unified detection-segmentation framework
/// - Production systems needing both detection boxes and segmentation masks
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Built on DINO detector with deformable attention transformer encoder-decoder
/// - Adds a mask branch using dot product between query embeddings and pixel embeddings
/// - Unified query matching for both box and mask predictions via Hungarian matching
/// - Backbone: ResNet-50 or Swin-L
/// - Achieves 54.5 AP on COCO instance, 59.4 PQ on COCO panoptic
/// </para>
/// <para>
/// <b>Reference:</b> Li et al., "Mask DINO: Towards A Unified Transformer-based Framework
/// for Object Detection and Segmentation", CVPR 2023.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a MaskDINO model for joint detection and segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 512, inputWidth: 512, inputDepth: 3, outputSize: 80);
/// var model = new MaskDINO&lt;double&gt;(architecture, numClasses: 80);
///
/// // Or load a pre-trained ONNX model for unified detection-segmentation
/// var onnxModel = new MaskDINO&lt;double&gt;(architecture, "maskdino.onnx", numClasses: 80);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation", "https://arxiv.org/abs/2206.02777", Year = 2023, Authors = "Feng Li, Hao Zhang, Huaizhe Xu, Shilong Liu, Lei Zhang, Lionel M. Ni, Heung-Yeung Shum")]
public partial class MaskDINO<T> : Common.PanopticSegmentationBase<T>
{
    private readonly MaskDINOOptions _options;

    /// <summary>
    /// Gets the configuration options for this Mask DINO model.
    /// </summary>
    /// <returns>The <see cref="MaskDINOOptions"/> for this model instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Options control model behavior including random seed for reproducibility.
    /// </para>
    /// </remarks>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    // Only Mask DINO's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from PanopticSegmentationBase -> SegmentationModelBase.
    private readonly int _numQueries;
    private readonly MaskDINOModelSize _modelSize;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;

    #endregion

    #region Properties

    // SupportsTraining, NumClasses, InputHeight, InputWidth and IsOnnxMode are inherited from
    // SegmentationModelBase and say exactly the same thing.
    internal bool UseNativeMode => _useNativeMode;
    internal MaskDINOModelSize ModelSize => _modelSize;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes Mask DINO in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW as specified in the paper).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of output classes (default: 80 for COCO).</param>
    /// <param name="numQueries">Number of object queries (default: 300, as in the paper).</param>
    /// <param name="modelSize">Backbone size (default: R50).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable Mask DINO model. 300 queries means the model
    /// can detect and segment up to 300 objects per image. The unified architecture jointly
    /// optimizes both detection boxes and segmentation masks.
    /// </para>
    /// </remarks>
    public MaskDINO(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numClasses = 80,
        int numQueries = 300,
        MaskDINOModelSize modelSize = MaskDINOModelSize.R50,
        double dropRate = 0.1,
        MaskDINOOptions? options = null)
        // The base resolves numClasses/native-mode/optimizer. `optimizer` is passed straight through,
        // INCLUDING null: the base defaults it lazily via CreateDefaultOptimizer(), which is the one
        // thing `optimizer ?? new AdamWOptimizer<...>(this)` could never do in a constructor initializer.
        : base(architecture, optimizer, lossFunction, numClasses,
               Math.Max(1, numClasses / 3), numClasses - Math.Max(1, numClasses / 3))
    {
        _options = options ?? new MaskDINOOptions();
        Options = _options;
        // Mask DINO's own detection-scale defaults, which differ from the base's 512x512.
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 800;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1333;
        _numQueries = numQueries;
        _modelSize = modelSize;
        _dropRate = dropRate;

        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes Mask DINO in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of output classes (default: 80).</param>
    /// <param name="numQueries">Number of object queries (default: 300).</param>
    /// <param name="modelSize">Backbone size for metadata (default: R50).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained Mask DINO from ONNX for fast inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public MaskDINO(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 80,
        int numQueries = 300,
        MaskDINOModelSize modelSize = MaskDINOModelSize.R50,
        MaskDINOOptions? options = null)
        // The base's ONNX constructor already validates the path, sets ONNX mode, resolves the input
        // geometry and opens the InferenceSession - the same twenty lines this used to repeat.
        : base(architecture, onnxModelPath, numClasses,
               Math.Max(1, numClasses / 3), numClasses - Math.Max(1, numClasses / 3))
    {
        _options = options ?? new MaskDINOOptions();
        Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 800;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1333;
        _numQueries = numQueries;
        _modelSize = modelSize;
        _dropRate = 0.0;

        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Runs a forward pass to produce detection boxes and segmentation masks.
    /// </summary>
    /// <param name="input">The input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Per-pixel segmentation logits tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass an image to get joint detection and segmentation predictions.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        return _useNativeMode ? Forward(input) : PredictOnnx(input);
    }

    /// <summary>
    /// Performs one training step with forward, loss, backward, and parameter update.
    /// </summary>
    /// <param name="input">The input image tensor.</param>
    /// <param name="expectedOutput">Ground-truth segmentation tensor.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Trains the model by comparing predictions to correct answers.
    /// Only available in native mode.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when called on an ONNX-mode model.</exception>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException(
                "Training is not supported in ONNX mode. Use the native mode constructor for training.");

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

    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(MaskDINOModelSize modelSize)
    {
        return modelSize switch
        {
            MaskDINOModelSize.R50 => ([256, 512, 1024, 2048], [3, 4, 6, 3], 256),
            MaskDINOModelSize.SwinLarge => ([192, 384, 768, 1536], [2, 2, 18, 2], 256),
            _ => ([256, 512, 1024, 2048], [3, 4, 6, 3], 256)
        };
    }

    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);

        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++)
            features = Layers[i].Forward(features);
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            features = Layers[i].Forward(features);

        if (!hasBatch) features = RemoveBatchDimension(features);
        return features;
    }

    protected override Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
            inputData[i] = Convert.ToSingle(input.Data.Span[i]);

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "pixel_values";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };

        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var outputShape = outputTensor.Dimensions.ToArray();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));

        var result = new Tensor<T>(outputShape, new Vector<T>(outputData));
        if (!hasBatch) result = RemoveBatchDimension(result);
        return result;
    }

    // AddBatchDimension and RemoveBatchDimension come from SegmentationModelBase.

    #endregion

    #region Abstract Implementation

    /// <summary>
    /// Initializes the encoder and decoder layers for Mask DINO.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In native mode, builds the backbone encoder (ResNet/Swin), deformable
    /// transformer encoder-decoder, and mask prediction head. In ONNX mode, layers are skipped.
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
            var encoderLayers = LayerHelper<T>.CreateMaskDINOEncoderLayers(
                _channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count;
            Layers.AddRange(encoderLayers);

            int featureH = _height / 32;
            int featureW = _width / 32;
            int encoderOutputChannels = _channelDims[^1];
            var decoderLayers = LayerHelper<T>.CreateMaskDINODecoderLayers(
                encoderOutputChannels, _decoderDim, _numClasses, featureH, featureW);
            Layers.AddRange(decoderLayers);
        }
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <summary>
    /// Collects metadata describing this Mask DINO model's configuration.
    /// </summary>
    /// <returns>Model metadata including type, architecture, and serialized data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns a summary for saving or display.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "MaskDINO" }, { "InputHeight", _height }, { "InputWidth", _width },
                { "InputChannels", _channels }, { "NumClasses", _numClasses },
                { "NumQueries", _numQueries }, { "ModelSize", _modelSize.ToString() },
                { "DecoderDim", _decoderDim }, { "UseNativeMode", _useNativeMode },
                { "NumLayers", Layers.Count }
            },
            ModelData = SerializeForMetadata()
        };
    }

    // Dispose is inherited: SegmentationModelBase already disposes _onnxSession and sets _disposed,
    // and Mask DINO owns no further unmanaged resources.

    #endregion

    #region IPanopticSegmentation Implementation

    // NumClasses, InputHeight, InputWidth, IsOnnxMode, Segment, NumStuffClasses and NumThingClasses
    // are all supplied by SegmentationModelBase / PanopticSegmentationBase.

    /// <inheritdoc/>
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
