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
/// YOLOv9-Seg: Instance segmentation with Programmable Gradient Information.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> YOLOv9-Seg extends the YOLOv9 object detector with a mask prediction
/// branch for real-time instance segmentation. It uses Programmable Gradient Information (PGI)
/// to preserve important information during training and GELAN (Generalized ELAN) for efficient
/// feature aggregation.
///
/// Common use cases:
/// - Real-time instance segmentation in video streams
/// - Autonomous driving (detecting and masking cars, pedestrians)
/// - Industrial inspection (defect detection with masks)
/// - Robotics (object manipulation with precise boundaries)
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Programmable Gradient Information (PGI) prevents information loss in deep networks
/// - GELAN (Generalized ELAN) architecture for efficient feature aggregation
/// - Achieves 43.5 mask mAP on COCO with real-time speed
/// - Anchor-free detection head + YOLACT-style mask branch
/// </para>
/// <para>
/// <b>Reference:</b> Wang et al., "YOLOv9: Learning What You Want to Learn Using Programmable
/// Gradient Information", arXiv 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a YOLOv9-Seg model with programmable gradient information
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 640, inputWidth: 640, inputDepth: 3, outputSize: 80);
/// var model = new YOLOv9Seg&lt;double&gt;(architecture, numClasses: 80);
///
/// // Or load a pre-trained ONNX model for industrial inspection
/// var onnxModel = new YOLOv9Seg&lt;double&gt;(architecture, "yolov9c-seg.onnx", numClasses: 80);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Segmentation)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information", "https://arxiv.org/abs/2402.13616", Year = 2024, Authors = "Wang et al.")]
public partial class YOLOv9Seg<T> : Common.InstanceSegmentationBase<T>
{
    private readonly YOLOv9SegOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    // Only YOLOv9Seg's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from InstanceSegmentationBase -> SegmentationModelBase, as do MaxInstances,
    // ConfidenceThreshold and NmsThreshold.
    private readonly YOLOv9SegModelSize _modelSize;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    #endregion

    #region Properties
    // SupportsTraining, NumClasses, InputHeight, InputWidth and IsOnnxMode are inherited from
    // SegmentationModelBase and say exactly the same thing.
    internal bool UseNativeMode => _useNativeMode;
    internal YOLOv9SegModelSize ModelSize => _modelSize;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes YOLOv9-Seg in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of object classes (default: 80 for COCO).</param>
    /// <param name="modelSize">Model size variant (default: C).</param>
    /// <param name="dropRate">Dropout rate (default: 0.0 — YOLO models rarely use dropout).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable YOLOv9-Seg. The PGI mechanism ensures gradient
    /// information flows correctly through the deep network during training.
    /// </para>
    /// </remarks>
    public YOLOv9Seg(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 80,
        YOLOv9SegModelSize modelSize = YOLOv9SegModelSize.C, double dropRate = 0.0,
        YOLOv9SegOptions? options = null)
        // The base resolves height/width/channels/numClasses/native-mode from the architecture and
        // defaults the loss to CrossEntropyWithLogitsLoss - exactly what the deleted lines did by
        // hand. `optimizer` is passed straight through INCLUDING null; the base's lazy
        // CreateDefaultOptimizer() produces the same `new AdamWOptimizer<...>(this)` default, which
        // could never be written as a base-constructor argument because `this` is unavailable there.
        : base(architecture, optimizer, lossFunction, numClasses)
    {
        _options = options ?? new YOLOv9SegOptions(); Options = _options;
        ApplyYoloInputFallback(architecture);
        _modelSize = modelSize; _dropRate = dropRate;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);
        InitializeLayers();
    }

    /// <summary>
    /// Re-applies YOLOv9-Seg's 640x640 fallback for architectures that carry no input geometry.
    /// </summary>
    /// <remarks>
    /// <para>
    /// SegmentationModelBase falls back to 512x512 when the architecture supplies no input height
    /// or width. Every YOLO variant is trained and exported at 640x640, so that fallback is
    /// restored here for the unset case only - when the architecture does specify dimensions, the
    /// base's value already matches and nothing changes.
    /// </para>
    /// </remarks>
    private void ApplyYoloInputFallback(NeuralNetworkArchitecture<T> architecture)
    {
        if (architecture.InputHeight <= 0) _height = 640;
        if (architecture.InputWidth <= 0) _width = 640;
    }

    /// <summary>
    /// Initializes YOLOv9-Seg in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of object classes (default: 80).</param>
    /// <param name="modelSize">Model size for metadata (default: C).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained YOLOv9-Seg from ONNX for real-time inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public YOLOv9Seg(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 80, YOLOv9SegModelSize modelSize = YOLOv9SegModelSize.C,
        YOLOv9SegOptions? options = null)
        // The base's ONNX constructor already validates the path, sets ONNX mode, resolves the input
        // geometry and opens the InferenceSession - the same lines this used to repeat.
        : base(architecture, onnxModelPath, numClasses)
    {
        _options = options ?? new YOLOv9SegOptions(); Options = _options;
        ApplyYoloInputFallback(architecture);
        _modelSize = modelSize; _dropRate = 0.0;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);
        InitializeLayers();
    }
    #endregion

    #region Public Methods
    // PredictCore is inherited from SegmentationModelBase and dispatches to Forward / PredictOnnx
    // exactly as the deleted override did.

    /// <summary>
    /// Performs one training step.
    /// </summary>
    /// <param name="input">The input image tensor.</param>
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
    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(YOLOv9SegModelSize modelSize) => modelSize switch
    {
        YOLOv9SegModelSize.C => ([64, 128, 256, 512], [3, 6, 6, 3], 256),
        YOLOv9SegModelSize.E => ([80, 160, 320, 640], [4, 8, 8, 4], 256),
        _ => ([64, 128, 256, 512], [3, 6, 6, 3], 256)
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

    // AddBatchDimension and RemoveBatchDimension are inherited from SegmentationModelBase.
    #endregion

    #region Abstract Implementation
    /// <summary>
    /// Initializes the backbone and detection/segmentation head layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In native mode, builds the GELAN backbone and mask prediction head.
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
            var encoderLayers = LayerHelper<T>.CreateYOLOv9SegEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateYOLOv9SegDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "YOLOv9Seg" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "ModelSize", _modelSize.ToString() }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
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
        ? new YOLOv9Seg<T>(Architecture, Optimizer, LossFunction, _numClasses, _modelSize, _dropRate, _options)
        : new YOLOv9Seg<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _modelSize, _options);

    // Dispose is inherited from SegmentationModelBase, which already disposes the ONNX session.
    // YOLOv9Seg owns no further unmanaged resources.
    #endregion

    #region IInstanceSegmentation Implementation

    // NumClasses, InputHeight, InputWidth, IsOnnxMode and Segment come from SegmentationModelBase;
    // MaxInstances (default 100), ConfidenceThreshold (0.5) and NmsThreshold (0.5) come from
    // InstanceSegmentationBase with the same defaults these explicit implementations hard-coded.

    /// <inheritdoc/>
    public override InstanceSegmentationResult<T> DetectInstances(Tensor<T> image)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var logits = Predict(image);
        var probMap = Common.SegmentationTensorOps.SoftmaxAlongClassDim(logits);
        var classMap = Common.SegmentationTensorOps.ArgmaxAlongClassDim(logits);
        int h = classMap.Shape[0], w = classMap.Shape[1];
        var instances = new List<InstanceMask<T>>();

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
                    for (int col = 0; col < w; col++)
                        if (NumOps.Compare(labelMap[row, col], NumOps.FromDouble(comp)) == 0)
                        {
                            mask[row, col] = NumOps.FromDouble(1.0);
                            area++;
                            sumConf += NumOps.ToDouble(probMap[cls, row, col]);
                            if (col < minX) minX = col; if (col > maxX) maxX = col;
                            if (row < minY) minY = row; if (row > maxY) maxY = row;
                        }
                if (area < 4) continue;
                double confidence = sumConf / area;
                if (confidence < ConfidenceThreshold) continue;
                var box = new BoundingBox<T>(NumOps.FromDouble(minX), NumOps.FromDouble(minY),
                    NumOps.FromDouble(maxX + 1), NumOps.FromDouble(maxY + 1), BoundingBoxFormat.XYXY, cls);
                instances.Add(new InstanceMask<T>(box, mask, cls, NumOps.FromDouble(confidence)));
            }
        }

        instances = instances.OrderByDescending(i => NumOps.ToDouble(i.Confidence)).ToList();
        var kept = new List<InstanceMask<T>>();
        while (instances.Count > 0)
        {
            var best = instances[0]; kept.Add(best); instances.RemoveAt(0);
            instances = instances.Where(inst => best.ComputeMaskIoU(inst, NumOps) < NmsThreshold).ToList();
        }

        sw.Stop();
        return new InstanceSegmentationResult<T> { Instances = kept, ImageHeight = h, ImageWidth = w, InferenceTime = sw.Elapsed };
    }

    #endregion
}
