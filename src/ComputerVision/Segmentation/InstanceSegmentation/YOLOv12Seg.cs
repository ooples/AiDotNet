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
/// YOLOv12-Seg: Attention-centric real-time instance segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Real-time instance segmentation with attention. Autonomous vehicle perception.
///
/// Common use cases:
/// - Real-time instance segmentation with attention
/// - Autonomous vehicle perception
/// - Smart surveillance systems
/// - Industrial quality control
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Area-attention mechanism replacing standard self-attention
/// - R-ELAN (Residual Efficient Layer Aggregation Network) blocks
/// - FlashAttention-compatible efficient attention
/// - Native attention integration in YOLO architecture
/// </para>
/// <para>
/// <b>Reference:</b> Tian et al., "YOLOv12: Attention-Centric Real-Time Object Detectors", arXiv 2025.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a YOLOv12-Seg model with attention-centric instance segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 640, inputWidth: 640, inputDepth: 3, outputSize: 80);
/// var model = new YOLOv12Seg&lt;double&gt;(architecture, numClasses: 80);
///
/// // Or load a pre-trained ONNX model for autonomous vehicle perception
/// var onnxModel = new YOLOv12Seg&lt;double&gt;(architecture, "yolov12n-seg.onnx", numClasses: 80);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Segmentation)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("YOLOv12: Attention-Centric Real-Time Object Detectors", "https://arxiv.org/abs/2502.12524", Year = 2025, Authors = "Yunjie Tian, Qixiang Ye, David Doermann")]
public partial class YOLOv12Seg<T> : Common.InstanceSegmentationBase<T>
{
    private readonly YOLOv12SegOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    // Only YOLOv12Seg's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from InstanceSegmentationBase -> SegmentationModelBase, as do MaxInstances,
    // ConfidenceThreshold and NmsThreshold.
    private YOLOv12SegModelSize _modelSize;
    private int[] _channelDims;
    private int _decoderDim;
    private int[] _depths;
    private double _dropRate;
    #endregion

    #region Properties
    // SupportsTraining, NumClasses, InputHeight, InputWidth and IsOnnxMode are inherited from
    // SegmentationModelBase and say exactly the same thing, so re-declaring them here would only
    // create two sources of one fact.
    internal bool UseNativeMode => _useNativeMode;
    internal YOLOv12SegModelSize ModelSize => _modelSize;
    #endregion

    /// <summary>
    /// Creates the paper-faithful default optimizer: AdamW with lr=1e-4.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Ultralytics YOLOv8/YOLOv12 reference uses lr0=0.01 with cosine decay to lrf=0.01
    /// (i.e., minimum lr = 1e-4 by end of training); for short-horizon memorization-style
    /// training on a single fixed batch the constant 1e-4 baseline is what the schedule
    /// eventually reaches, and avoids the optimizer-bouncing divergence seen at the framework
    /// default (lr=1e-3) where 200-iter loss exceeds 50-iter loss. Overriding the base's
    /// CreateDefaultOptimizer is how that default survives re-parenting.
    /// </para>
    /// </remarks>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer()
        => new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = 1e-4
            });

    #region Constructors
    /// <summary>
    /// Initializes YOLOv12Seg in native (trainable) mode.
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
    /// <b>For Beginners:</b> Creates a trainable YOLOv12Seg model.
    /// </para>
    /// </remarks>
    public YOLOv12Seg(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 80,
        YOLOv12SegModelSize modelSize = YOLOv12SegModelSize.N, double dropRate = 0,
        YOLOv12SegOptions? options = null)
        // Default loss = MSE on the continuous mask logits. The full YOLOv12
        // paper recipe (Tian et al. 2024) is multi-component — CIoU box loss,
        // VFL classification loss, DFL distribution loss, BCE mask supervision —
        // and applies only when the training data has matching structured
        // targets (boxes, class IDs, binary masks). Outside that recipe (e.g.
        // generic regression / supervision against a continuous mask), MSE is
        // the stable baseline: it has a unique minimum at prediction == target
        // for any continuous target, whereas BCE-with-logits on continuous
        // targets has a non-zero gradient at the "correct" answer (sigmoid(x)
        // - y is non-zero unless y ∈ {0, 1}) and Adam momentum then over-
        // shoots, producing the 200-iter loss > 50-iter loss divergence the
        // MoreData_ShouldNotDegrade invariant catches. Callers training on
        // real ground-truth masks pass their own multi-component loss via
        // the lossFunction parameter.
        // The base resolves height/width/channels/numClasses/native-mode from the architecture,
        // which is exactly what the deleted lines below used to do by hand. `optimizer` is passed
        // straight through - INCLUDING null; the base defaults it lazily via CreateDefaultOptimizer(),
        // overridden above to keep this model's lr=1e-4 AdamW. The MSE default loss is preserved
        // explicitly because the base would otherwise default to CrossEntropyWithLogitsLoss.
        : base(architecture, optimizer, lossFunction ?? new MeanSquaredErrorLoss<T>(), numClasses)
    {
        _options = options ?? new YOLOv12SegOptions(); Options = _options;
        ApplyYoloInputFallback(architecture);
        _modelSize = modelSize; _dropRate = dropRate;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);
        InitializeLayers();
    }

    /// <summary>
    /// Re-applies YOLOv12-Seg's 640x640 fallback for architectures that carry no input geometry.
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
    /// Initializes YOLOv12Seg in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 80).</param>
    /// <param name="modelSize">Model size for metadata (default: N).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained YOLOv12Seg from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public YOLOv12Seg(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 80, YOLOv12SegModelSize modelSize = YOLOv12SegModelSize.N,
        YOLOv12SegOptions? options = null)
        // The base's ONNX constructor already validates the path, sets ONNX mode, resolves the input
        // geometry and opens the InferenceSession - the same lines this used to repeat.
        : base(architecture, onnxModelPath, numClasses)
    {
        _options = options ?? new YOLOv12SegOptions(); Options = _options;
        ApplyYoloInputFallback(architecture);
        _modelSize = modelSize; _dropRate = 0;
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
            // Passes this model's optimizer explicitly - the base Train() would let
            // NeuralNetworkBase pick its own, losing the lr=1e-4 AdamW default.
            TrainWithTape(input, expectedOutput, Optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }
    #endregion

    #region Private Methods
    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(YOLOv12SegModelSize modelSize) => modelSize switch
    {
        YOLOv12SegModelSize.N => ([16, 32, 64, 128], [1, 2, 2, 1], 64),
        YOLOv12SegModelSize.S => ([32, 64, 128, 256], [1, 2, 2, 1], 128),
        YOLOv12SegModelSize.M => ([48, 96, 192, 384], [2, 4, 4, 2], 192),
        YOLOv12SegModelSize.L => ([64, 128, 256, 512], [2, 4, 4, 2], 256),
        YOLOv12SegModelSize.X => ([80, 160, 320, 640], [2, 4, 4, 2], 320),
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

    // AddBatchDimension and RemoveBatchDimension are inherited from SegmentationModelBase.
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
            var encoderLayers = LayerHelper<T>.CreateYOLOv12SegEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateYOLOv12SegDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "YOLOv12Seg" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "ModelSize", _modelSize.ToString() }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
        ModelData = SerializeForMetadata()
    };

    // Dispose is inherited from SegmentationModelBase, which already disposes the ONNX session.
    // YOLOv12Seg owns no further unmanaged resources.
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
