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
/// MixedQueryTransformer (MQ-Former): Dynamic Query Melding for Multi-Dataset Segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> MixedQueryTransformer scales mask-based segmentation across multiple diverse datasets
/// by dynamically melding (fusing) instance queries and stuff queries through cross-attention. This
/// allows the model to generalize well across different segmentation benchmarks without dataset-specific
/// fine-tuning.
///
/// Common use cases:
/// - Multi-dataset panoptic segmentation
/// - Cross-domain segmentation transfer
/// - Production systems trained on diverse data sources
/// - Research in universal segmentation scaling
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Dynamic query melding: instance and stuff queries interact via cross-attention layers
/// - Multi-dataset training with unified query representations
/// - Backbone: ResNet-50 or Swin-L transformer
/// - Built on Mask2Former architecture with query interaction extensions
/// </para>
/// <para>
/// <b>Reference:</b> "MixedQueryTransformer: Dynamic Query Melding for Multi-Dataset Segmentation", CVPR 2025.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a MixedQueryTransformer model for multi-dataset panoptic segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 512, inputWidth: 512, inputDepth: 3, outputSize: 133);
/// var model = new MixedQueryTransformer&lt;double&gt;(architecture, numClasses: 133);
///
/// // Or load a pre-trained ONNX model for cross-dataset segmentation
/// var onnxModel = new MixedQueryTransformer&lt;double&gt;(architecture, "querymeldnet.onnx", numClasses: 133);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ResearchPaper("Mixed-Query Transformer: A Unified Image Segmentation Architecture",
    "https://arxiv.org/abs/2404.04469",
    Year = 2024,
    Authors = "Pei Wang, Zhaowei Cai, Hao Yang, Ashwin Swaminathan, R. Manmatha, Stefano Soatto")]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
public class MixedQueryTransformer<T> : Common.PanopticSegmentationBase<T>
{
    private readonly MixedQueryTransformerOptions _options;

    /// <summary>
    /// Gets the configuration options for this MixedQueryTransformer model.
    /// </summary>
    /// <returns>The <see cref="MixedQueryTransformerOptions"/> for this model instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Options control model behavior including random seed for reproducibility.
    /// </para>
    /// </remarks>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    // Only MixedQueryTransformer's OWN configuration lives here. _height, _width, _channels,
    // _numClasses, _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and
    // _encoderLayerEnd all come from PanopticSegmentationBase -> SegmentationModelBase.
    private readonly int _numQueries;
    private readonly MixedQueryTransformerModelSize _modelSize;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;

    #endregion

    #region Properties

    // SupportsTraining, NumClasses, InputHeight, InputWidth and IsOnnxMode are inherited from
    // SegmentationModelBase and say exactly the same thing.
    internal bool UseNativeMode => _useNativeMode;
    internal MixedQueryTransformerModelSize ModelSize => _modelSize;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes MixedQueryTransformer in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of output classes (default: 133 for COCO panoptic).</param>
    /// <param name="numQueries">Number of melded queries (default: 200).</param>
    /// <param name="modelSize">Backbone size (default: R50).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable MixedQueryTransformer. The 200 queries are split between
    /// instance queries (for countable objects) and stuff queries (for uncountable regions like sky),
    /// and then melded via cross-attention for improved multi-dataset performance.
    /// </para>
    /// </remarks>
    public MixedQueryTransformer(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numClasses = 133,
        int numQueries = 200,
        MixedQueryTransformerModelSize modelSize = MixedQueryTransformerModelSize.R50,
        double dropRate = 0.1,
        MixedQueryTransformerOptions? options = null)
        // The base resolves height/width/channels/numClasses/native-mode from the architecture, and
        // defaults `optimizer` LAZILY via CreateDefaultOptimizer() - which is why null is passed
        // straight through instead of `optimizer ?? new AdamWOptimizer<...>(this)`, an expression
        // that cannot appear in a constructor initializer.
        : base(architecture, optimizer, lossFunction, numClasses,
               Math.Max(1, numClasses / 3), numClasses - Math.Max(1, numClasses / 3))
    {
        _options = options ?? new MixedQueryTransformerOptions();
        Options = _options;
        _numQueries = numQueries;
        _modelSize = modelSize;
        _dropRate = dropRate;

        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);
        InitializeLayers();
    }

    /// <inheritdoc />
    /// <remarks>
    /// THE PAPER'S RATE, NOT THE LIBRARY DEFAULT. Constructed with no options this trained at
    /// InitialLearningRate = 1e-3, ten times the published rate and unreachable by a caller.
    /// <para>
    /// This lives in the hook rather than in the constructor because the base builds the optimizer
    /// LAZILY - assigning the field directly in the constructor is what the re-parenting removed, and
    /// re-adding it here would have been overwritten the first time the base resolved its own. The
    /// rate itself is preserved exactly; only where it is applied has moved.
    /// </para>
    /// </remarks>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer()
        => new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
            });

    /// <summary>
    /// Initializes MixedQueryTransformer in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of output classes (default: 133).</param>
    /// <param name="numQueries">Number of melded queries (default: 200).</param>
    /// <param name="modelSize">Backbone size for metadata (default: R50).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained MixedQueryTransformer from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public MixedQueryTransformer(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 133,
        int numQueries = 200,
        MixedQueryTransformerModelSize modelSize = MixedQueryTransformerModelSize.R50,
        MixedQueryTransformerOptions? options = null)
        // The base's ONNX constructor already validates the path, sets ONNX mode, resolves the input
        // geometry and opens the InferenceSession - the same twenty lines this used to repeat.
        : base(architecture, onnxModelPath, numClasses,
               Math.Max(1, numClasses / 3), numClasses - Math.Max(1, numClasses / 3))
    {
        _options = options ?? new MixedQueryTransformerOptions();
        Options = _options;
        _numQueries = numQueries;
        _modelSize = modelSize;
        _dropRate = 0.0;

        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Runs a forward pass through MixedQueryTransformer for multi-dataset segmentation.
    /// </summary>
    /// <param name="input">The input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Per-pixel segmentation logits tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass an image to get segmentation predictions with melded queries.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        return _useNativeMode ? Forward(input) : PredictOnnx(input);
    }

    /// <summary>
    /// Performs one training step.
    /// </summary>
    /// <param name="input">The input image tensor.</param>
    /// <param name="expectedOutput">Ground-truth segmentation tensor.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Trains using multi-dataset joint optimization. Only native mode.
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

    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(MixedQueryTransformerModelSize modelSize)
    {
        return modelSize switch
        {
            MixedQueryTransformerModelSize.R50 => ([256, 512, 1024, 2048], [3, 4, 6, 3], 256),
            MixedQueryTransformerModelSize.SwinLarge => ([192, 384, 768, 1536], [2, 2, 18, 2], 256),
            _ => ([256, 512, 1024, 2048], [3, 4, 6, 3], 256)
        };
    }

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
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };
        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++) outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        var result = new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
        if (!hasBatch) result = RemoveBatchDimension(result);
        return result;
    }

    // AddBatchDimension and RemoveBatchDimension come from SegmentationModelBase.

    #endregion

    #region Abstract Implementation

    /// <summary>
    /// Initializes the encoder and decoder layers for MixedQueryTransformer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In native mode, builds the backbone encoder and query-meld decoder.
    /// In ONNX mode, no layers are created.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) { ClearLayers(); return; }
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = _options.EncoderLayerCount ?? Architecture.Layers.Count / 2;
        }
        else
        {
            var encoderLayers = LayerHelper<T>.CreateMixedQueryTransformerEncoderLayers(
                _channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count;
            Layers.AddRange(encoderLayers);
            int featureH = _height / 32;
            int featureW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateMixedQueryTransformerDecoderLayers(
                _channelDims[^1], _decoderDim, _numClasses, featureH, featureW);
            Layers.AddRange(decoderLayers);
        }
    }

    // UpdateParameters folded one enumeration the base already folds. Removed under AIDN082.
    /// <summary>
    /// Collects metadata describing this MixedQueryTransformer model's configuration.
    /// </summary>
    /// <returns>Model metadata.</returns>
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
                { "ModelName", "MixedQueryTransformer" }, { "InputHeight", _height }, { "InputWidth", _width },
                { "InputChannels", _channels }, { "NumClasses", _numClasses },
                { "NumQueries", _numQueries }, { "ModelSize", _modelSize.ToString() },
                { "DecoderDim", _decoderDim }, { "UseNativeMode", _useNativeMode },
                { "NumLayers", Layers.Count }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <summary>
    /// Writes MixedQueryTransformer configuration to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Saves model configuration for later reconstruction.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_height); writer.Write(_width); writer.Write(_channels);
        writer.Write(_numClasses); writer.Write(_numQueries); writer.Write((int)_modelSize);
        writer.Write(_decoderDim); writer.Write(_dropRate);
        writer.Write(_useNativeMode); writer.Write(_onnxModelPath ?? string.Empty);
        writer.Write(_encoderLayerEnd);
        writer.Write(_channelDims.Length);
        foreach (int dim in _channelDims) writer.Write(dim);
        writer.Write(_depths.Length);
        foreach (int depth in _depths) writer.Write(depth);
    }

    /// <summary>
    /// Reads MixedQueryTransformer configuration from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads model configuration when restoring a saved model.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32();
        _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32();
        _ = reader.ReadInt32(); _ = reader.ReadDouble();
        _ = reader.ReadBoolean(); _ = reader.ReadString(); _ = reader.ReadInt32();
        int dimCount = reader.ReadInt32();
        for (int i = 0; i < dimCount; i++) _ = reader.ReadInt32();
        int depthCount = reader.ReadInt32();
        for (int i = 0; i < depthCount; i++) _ = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new MixedQueryTransformer instance with the same configuration but fresh weights.
    /// </summary>
    /// <returns>A new <see cref="MixedQueryTransformer{T}"/> model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a copy for cross-validation or ensemble training.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return _useNativeMode
            ? new MixedQueryTransformer<T>(Architecture, Optimizer, LossFunction, _numClasses, _numQueries, _modelSize, _dropRate, _options)
            : new MixedQueryTransformer<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _numQueries, _modelSize, _options);
    }

    // Dispose is inherited: SegmentationModelBase already disposes _onnxSession and sets _disposed,
    // and MixedQueryTransformer owns no further unmanaged resources.

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
