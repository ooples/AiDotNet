using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Augmentation.Image;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;

/// <summary>
/// YOLO11-Seg: Ultralytics next-generation real-time instance segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Real-time instance segmentation. Edge deployment with INT8 quantization.
///
/// Common use cases:
/// - Real-time instance segmentation
/// - Edge deployment with INT8 quantization
/// - Video analytics pipelines
/// - Mobile and embedded applications
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - C2PSA (Cross-Stage Partial with Spatial Attention) blocks
/// - Improved feature pyramid for multi-scale detection
/// - Anchor-free decoupled head with mask branch
/// - YOLACT-style prototype mask generation
/// </para>
/// <para>
/// <b>Reference:</b> Ultralytics, "YOLO11", 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a YOLO11-Seg model for real-time instance segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 640, inputWidth: 640, inputDepth: 3, outputSize: 80);
/// var model = new YOLO11Seg&lt;double&gt;(architecture, numClasses: 80);
///
/// // Or load a pre-trained ONNX model for edge deployment
/// var onnxModel = new YOLO11Seg&lt;double&gt;(architecture, "yolo11n-seg.onnx", numClasses: 80);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Segmentation)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.Medium)]
[ResearchPaper("Ultralytics YOLO11", "https://docs.ultralytics.com/models/yolo11/")]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
public partial class YOLO11Seg<T> : Common.InstanceSegmentationBase<T>
{
    private readonly YOLO11SegOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    // Only YOLO11Seg's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from InstanceSegmentationBase -> SegmentationModelBase, as do the detection knobs
    // MaxInstances / ConfidenceThreshold / NmsThreshold.
    private readonly YOLO11SegModelSize _modelSize;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    #endregion

    #region Properties
    // SupportsTraining, NumClasses, InputHeight, InputWidth and IsOnnxMode are inherited from
    // SegmentationModelBase and say exactly the same thing.
    internal bool UseNativeMode => _useNativeMode;
    internal YOLO11SegModelSize ModelSize => _modelSize;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes YOLO11Seg in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 80).</param>
    /// <param name="modelSize">Model size variant (default: N).</param>
    /// <param name="dropRate">Dropout rate (default: 0).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable YOLO11Seg model.
    /// </para>
    /// </remarks>
    public YOLO11Seg(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 80,
        YOLO11SegModelSize modelSize = YOLO11SegModelSize.N, double dropRate = 0,
        YOLO11SegOptions? options = null)
        // The base resolves numClasses/native-mode plus the maxInstances=100, confidenceThreshold=0.5
        // and nmsThreshold=0.5 defaults this model used to keep as private fields. `optimizer` is
        // passed straight through INCLUDING null - the base defaults it lazily via
        // CreateDefaultOptimizer(), which `optimizer ?? new AdamWOptimizer<...>(this)` could never do
        // in a constructor initializer.
        : base(architecture, optimizer, lossFunction, numClasses)
    {
        _options = options ?? new YOLO11SegOptions(); Options = _options;
        // YOLO11Seg's own 640x640 input default, which differs from the base's 512x512.
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 640;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 640;
        _modelSize = modelSize; _dropRate = dropRate;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes YOLO11Seg in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 80).</param>
    /// <param name="modelSize">Model size for metadata (default: N).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained YOLO11Seg from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public YOLO11Seg(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 80, YOLO11SegModelSize modelSize = YOLO11SegModelSize.N,
        YOLO11SegOptions? options = null)
        // The base's ONNX constructor already validates the path, sets ONNX mode, resolves the input
        // geometry and opens the InferenceSession - the same fifteen lines this used to repeat.
        : base(architecture, onnxModelPath, numClasses)
    {
        _options = options ?? new YOLO11SegOptions(); Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 640;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 640;
        _modelSize = modelSize; _dropRate = 0;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);
        InitializeLayers();
    }
    #endregion

    #region Public Methods
    /// <summary>
    /// Runs a forward pass to produce segmentation logits.
    /// </summary>
    /// <param name="input">The input tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Segmentation logits tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass an image to get a per-pixel class prediction map.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input) => _useNativeMode ? Forward(input) : PredictOnnx(input);

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
    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(YOLO11SegModelSize modelSize) => modelSize switch
    {
        YOLO11SegModelSize.N => ([16, 32, 64, 128], [1, 2, 2, 1], 64),
        YOLO11SegModelSize.S => ([32, 64, 128, 256], [1, 2, 2, 1], 128),
        YOLO11SegModelSize.M => ([48, 96, 192, 384], [2, 4, 4, 2], 192),
        YOLO11SegModelSize.L => ([64, 128, 256, 512], [2, 4, 4, 2], 256),
        YOLO11SegModelSize.X => ([80, 160, 320, 640], [2, 4, 4, 2], 320),
        _ => ([16, 32, 64, 128], [1, 2, 2, 1], 64)
    };

    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        var features = input;
        for (int i = 0; i < _encoderLayerEnd; i++) features = Layers[i].Forward(features);
        for (int i = _encoderLayerEnd; i < Layers.Count; i++) features = Layers[i].Forward(features);
        if (!hasBatch) features = RemoveBatchDimension(features); return features;
    }

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

    // AddBatchDimension and RemoveBatchDimension come from SegmentationModelBase.
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
            var encoderLayers = LayerHelper<T>.CreateYOLO11SegEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateYOLO11SegDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "YOLO11Seg" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "ModelSize", _modelSize.ToString() }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
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
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => _useNativeMode
        ? new YOLO11Seg<T>(Architecture, Optimizer, LossFunction, _numClasses, _modelSize, _dropRate, _options)
        : new YOLO11Seg<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _modelSize, _options);

    // Dispose is inherited: SegmentationModelBase already disposes _onnxSession and sets _disposed,
    // and YOLO11Seg owns no further unmanaged resources.
    #endregion

    #region IInstanceSegmentation Implementation

    // NumClasses, InputHeight, InputWidth, IsOnnxMode and Segment come from SegmentationModelBase;
    // MaxInstances (100), ConfidenceThreshold (0.5) and NmsThreshold (0.5) come from
    // InstanceSegmentationBase with exactly these defaults, and are settable there too.

    /// <inheritdoc/>
    public override InstanceSegmentationResult<T> DetectInstances(Tensor<T> image)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var logits = Predict(image);
        var probMap = Common.SegmentationTensorOps.SoftmaxAlongClassDim(logits);
        var classMap = Common.SegmentationTensorOps.ArgmaxAlongClassDim(logits);
        int h = classMap.Shape[0], w = classMap.Shape[1];
        var instances = new List<InstanceMask<T>>();

        // Extract instances as connected components of each non-background class
        for (int cls = 1; cls < _numClasses && instances.Count < 100; cls++)
        {
            var (labelMap, count) = Common.SegmentationTensorOps.LabelConnectedComponents(classMap, cls);
            for (int comp = 1; comp <= count && instances.Count < 100; comp++)
            {
                var mask = new Tensor<T>([h, w]);
                int area = 0;
                double sumConf = 0;
                int minX = w, minY = h, maxX = 0, maxY = 0;

                for (int row = 0; row < h; row++)
                {
                    for (int col = 0; col < w; col++)
                    {
                        if (NumOps.Compare(labelMap[row, col], NumOps.FromDouble(comp)) == 0)
                        {
                            mask[row, col] = NumOps.FromDouble(1.0);
                            area++;
                            sumConf += NumOps.ToDouble(probMap[cls, row, col]);
                            if (col < minX) minX = col;
                            if (col > maxX) maxX = col;
                            if (row < minY) minY = row;
                            if (row > maxY) maxY = row;
                        }
                    }
                }

                if (area < 4) continue; // skip noise components
                double confidence = sumConf / area;
                if (confidence < ConfidenceThreshold) continue;

                var box = new BoundingBox<T>(
                    NumOps.FromDouble(minX), NumOps.FromDouble(minY),
                    NumOps.FromDouble(maxX + 1), NumOps.FromDouble(maxY + 1),
                    BoundingBoxFormat.XYXY, cls);
                instances.Add(new InstanceMask<T>(box, mask, cls, NumOps.FromDouble(confidence)));
            }
        }

        // Apply mask-based NMS to remove overlapping detections
        instances = instances.OrderByDescending(i => NumOps.ToDouble(i.Confidence)).ToList();
        var kept = new List<InstanceMask<T>>();
        while (instances.Count > 0)
        {
            var best = instances[0];
            kept.Add(best);
            instances.RemoveAt(0);
            instances = instances.Where(inst => best.ComputeMaskIoU(inst, NumOps) < NmsThreshold).ToList();
        }

        sw.Stop();
        return new InstanceSegmentationResult<T>
        {
            Instances = kept,
            ImageHeight = h,
            ImageWidth = w,
            InferenceTime = sw.Elapsed
        };
    }

    #endregion
}
