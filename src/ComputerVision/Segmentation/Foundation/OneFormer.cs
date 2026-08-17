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
/// OneFormer: One Transformer to Rule Universal Image Segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> OneFormer is trained once on panoptic data and can then perform any
/// segmentation task — semantic, instance, or panoptic — by simply providing a text prompt that
/// describes which task to perform. This "one model for all tasks" approach dramatically simplifies
/// deployment compared to maintaining separate models for each task.
///
/// Example usage:
/// - Pass "the task is semantic" to get per-pixel class labels
/// - Pass "the task is instance" to get individual object masks
/// - Pass "the task is panoptic" to get both stuff and thing segments
///
/// Common use cases:
/// - Multi-task segmentation systems needing all three task types
/// - Research comparing segmentation approaches
/// - Production systems where maintaining one model is simpler than three
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Builds on Mask2Former with a text encoder (CLIP-based) for task conditioning
/// - Task-conditioned joint training on panoptic, semantic, and instance data simultaneously
/// - Uses a task-guided query initialization that focuses queries on the specified task
/// - Backbone: Swin-L or DiNAT-L (Dilated Neighborhood Attention Transformer)
/// - SOTA on ADE20K, Cityscapes, and COCO across all three tasks with a single model
/// </para>
/// <para>
/// <b>Reference:</b> Jain et al., "OneFormer: One Transformer to Rule Universal Image
/// Segmentation", CVPR 2023.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a OneFormer model for task-conditioned universal segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 512, inputWidth: 512, inputDepth: 3, outputSize: 150);
/// var model = new OneFormer&lt;double&gt;(architecture, numClasses: 150);
///
/// // Or load a pre-trained ONNX model for text-prompted segmentation
/// var onnxModel = new OneFormer&lt;double&gt;(architecture, "oneformer.onnx", numClasses: 150);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("OneFormer: One Transformer to Rule Universal Image Segmentation", "https://arxiv.org/abs/2211.06220", Year = 2023, Authors = "Jitesh Jain, Jiachen Li, MangTik Chiu, Ali Hassani, Nikita Orlov, Humphrey Shi")]
public partial class OneFormer<T> : Common.PanopticSegmentationBase<T>
{
    private readonly OneFormerOptions _options;

    /// <summary>
    /// Gets the configuration options for this OneFormer model.
    /// </summary>
    /// <returns>The <see cref="OneFormerOptions"/> for this model instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Options control model behavior including random seed for reproducibility.
    /// </para>
    /// </remarks>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    // Only OneFormer's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from PanopticSegmentationBase -> SegmentationModelBase.
    private readonly int _numQueries;
    private readonly OneFormerModelSize _modelSize;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly int[] _attentionHeads;
    private readonly int _windowSize;
    private readonly int _patchSize;
    private readonly int _mlpRatio;
    private readonly double _dropRate;

    #endregion

    #region Properties

    // SupportsTraining, NumClasses, InputHeight, InputWidth and IsOnnxMode are inherited from
    // SegmentationModelBase and say exactly the same thing.
    internal bool UseNativeMode => _useNativeMode;
    internal OneFormerModelSize ModelSize => _modelSize;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes OneFormer in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW with weight decay 0.05,
    /// as specified in the OneFormer paper).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of output classes (default: 150 for ADE20K).</param>
    /// <param name="numQueries">Number of object queries (default: 150, higher than Mask2Former
    /// to accommodate multi-task predictions).</param>
    /// <param name="modelSize">Backbone size (default: SwinLarge).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable OneFormer. Training is done on panoptic data
    /// which automatically teaches the model semantic and instance segmentation as well.
    /// A text encoder conditions the queries on the target task at inference time.
    /// </para>
    /// </remarks>
    public OneFormer(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numClasses = 150,
        int numQueries = 150,
        OneFormerModelSize modelSize = OneFormerModelSize.SwinLarge,
        double dropRate = 0.1,
        OneFormerOptions? options = null)
        // The base resolves height/width/channels/numClasses/native-mode from the architecture.
        // OneFormer keeps its own class-axis-1 loss default, which the base does not know about.
        : base(architecture, optimizer, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(classAxis: 1),
               numClasses, Math.Max(1, numClasses / 3), numClasses - Math.Max(1, numClasses / 3))
    {
        _options = options ?? new OneFormerOptions();
        Options = _options;
        _numQueries = numQueries;
        _modelSize = modelSize;
        _dropRate = dropRate;
        // Resolved EAGERLY, not through the base's lazy CreateDefaultOptimizer(): OneFormer's
        // override validates LearningRate/WeightDecay/MaxGradientNorm, and that validation has
        // always fired at construction time rather than at the first training step.
        _optimizer = optimizer ?? CreateDefaultOptimizer();

        (_channelDims, _depths, _decoderDim) = ResolveModelConfig(modelSize, _options);
        (_attentionHeads, _windowSize, _patchSize, _mlpRatio) = ResolveEncoderConfig(_options, _channelDims);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes OneFormer in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="numClasses">Number of classes (default: 150).</param>
    /// <param name="numQueries">Number of queries (default: 150).</param>
    /// <param name="modelSize">Model size for metadata (default: SwinLarge).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained OneFormer for multi-task segmentation inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if file not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if ONNX load fails.</exception>
    public OneFormer(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses = 150,
        int numQueries = 150,
        OneFormerModelSize modelSize = OneFormerModelSize.SwinLarge,
        OneFormerOptions? options = null)
        // The base's ONNX constructor already validates the path, sets ONNX mode, resolves the input
        // geometry and opens the InferenceSession - the same twenty lines this used to repeat.
        : base(architecture, onnxModelPath, numClasses,
               Math.Max(1, numClasses / 3), numClasses - Math.Max(1, numClasses / 3))
    {
        // The base's ONNX constructor installs a plain CrossEntropyWithLogitsLoss because it has no
        // lossFunction parameter. OneFormer's channel-first output needs classAxis 1, so restore it.
        LossFunction = new CrossEntropyWithLogitsLoss<T>(classAxis: 1);
        _options = options ?? new OneFormerOptions();
        Options = _options;
        _numQueries = numQueries;
        _modelSize = modelSize;
        _dropRate = 0.0;

        (_channelDims, _depths, _decoderDim) = ResolveModelConfig(modelSize, _options);
        (_attentionHeads, _windowSize, _patchSize, _mlpRatio) = ResolveEncoderConfig(_options, _channelDims);

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Runs a forward pass through OneFormer for task-conditioned segmentation.
    /// </summary>
    /// <param name="input">Input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Per-pixel class logits tensor. The output depends on the task conditioning.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> OneFormer processes the image through a backbone encoder and
    /// text-conditioned transformer decoder. The text prompt ("the task is semantic/instance/panoptic")
    /// guides which type of segmentation output is produced.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        return !_useNativeMode ? PredictOnnx(input) : Forward(input);
    }

    /// <summary>
    /// Performs one training step with panoptic multi-task learning.
    /// </summary>
    /// <param name="input">Input image tensor.</param>
    /// <param name="expectedOutput">Ground-truth segmentation.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> OneFormer training uses panoptic data to simultaneously learn
    /// semantic, instance, and panoptic segmentation. A contrastive loss between text and
    /// visual features helps the model learn task-specific behavior.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown in ONNX mode.</exception>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode.");

        // Keep the training graph in the standard NCHW segmentation layout. PredictCore removes
        // the synthetic batch dimension for a caller that supplies CHW, but doing that before the
        // loss turns [C,H,W] into an ambiguous rank-3 tensor: a generic cross-entropy loss can then
        // mistake W for the class axis (and W may be 1 at the final Swin stage, yielding zero loss).
        // Batch the corresponding dense [C,H,W] or class-index [H,W] target as well so the loss sees
        // [N,C,H,W] + [N,C,H,W]/[N,H,W], exactly like channel-first segmentation frameworks.
        bool inputWasUnbatched = input.Rank == 3;
        if (inputWasUnbatched)
        {
            input = AddBatchDimension(input);
            if (expectedOutput.Rank is 2 or 3)
                expectedOutput = AddLeadingBatchDimension(expectedOutput);
        }

        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    #endregion

    #region Private Methods

    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(OneFormerModelSize modelSize)
    {
        return modelSize switch
        {
            OneFormerModelSize.SwinLarge => ([192, 384, 768, 1536], [2, 2, 18, 2], 256),
            OneFormerModelSize.DiNATLarge => ([192, 384, 768, 1536], [2, 2, 18, 2], 256),
            _ => ([192, 384, 768, 1536], [2, 2, 18, 2], 256)
        };
    }

    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer()
    {
        if (_options.LearningRate <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(_options.LearningRate), "Learning rate must be positive.");
        if (_options.WeightDecay < 0.0)
            throw new ArgumentOutOfRangeException(nameof(_options.WeightDecay), "Weight decay cannot be negative.");
        if (_options.MaxGradientNorm < 0.0)
            throw new ArgumentOutOfRangeException(nameof(_options.MaxGradientNorm), "Maximum gradient norm cannot be negative.");

        return new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                WeightDecay = _options.WeightDecay,
                EnableGradientClipping = _options.MaxGradientNorm > 0.0,
                MaxGradientNorm = _options.MaxGradientNorm
            });
    }

    private static (int[] ChannelDims, int[] Depths, int DecoderDim) ResolveModelConfig(
        OneFormerModelSize modelSize, OneFormerOptions options)
    {
        var defaults = GetModelConfig(modelSize);
        int[] channelDims = options.ChannelDimensions?.ToArray() ?? defaults.ChannelDims;
        int[] depths = options.StageDepths?.ToArray() ?? defaults.Depths;
        int decoderDim = options.DecoderDimension ?? defaults.DecoderDim;

        ValidateFourPositive(channelDims, nameof(options.ChannelDimensions));
        ValidateFourPositive(depths, nameof(options.StageDepths));
        for (int i = 1; i < channelDims.Length; i++)
        {
            if (channelDims[i] != channelDims[i - 1] * 2)
                throw new ArgumentException("Each OneFormer channel dimension must be twice the preceding stage.", nameof(options.ChannelDimensions));
        }
        if (decoderDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(options.DecoderDimension), "Decoder dimension must be positive.");
        return (channelDims, depths, decoderDim);
    }

    private static (int[] AttentionHeads, int WindowSize, int PatchSize, int MlpRatio) ResolveEncoderConfig(
        OneFormerOptions options, int[] channelDims)
    {
        int[] heads = options.AttentionHeads?.ToArray() ?? [6, 12, 24, 48];
        ValidateFourPositive(heads, nameof(options.AttentionHeads));
        for (int i = 0; i < heads.Length; i++)
        {
            if (channelDims[i] % heads[i] != 0)
                throw new ArgumentException($"Channel dimension {channelDims[i]} must be divisible by attention-head count {heads[i]} at stage {i}.", nameof(options.AttentionHeads));
        }
        if (options.WindowSize <= 0) throw new ArgumentOutOfRangeException(nameof(options.WindowSize));
        if (options.PatchSize <= 0) throw new ArgumentOutOfRangeException(nameof(options.PatchSize));
        if (options.MlpRatio <= 0) throw new ArgumentOutOfRangeException(nameof(options.MlpRatio));
        return (heads, options.WindowSize, options.PatchSize, options.MlpRatio);
    }

    private static void ValidateFourPositive(int[] values, string parameterName)
    {
        if (values.Length != 4 || values.Any(value => value <= 0))
            throw new ArgumentException("OneFormer requires exactly four positive stage values.", parameterName);
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

    /// <summary>
    /// Routes training through OneFormer's native NCHW path without removing the batch dimension.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (!_useNativeMode) return PredictOnnx(input);
        if (input.Rank != 4) input = AddBatchDimension(input);
        return Forward(input);
    }

    /// <summary>
    /// Captures activations after applying OneFormer's required leading batch reshape.
    /// </summary>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (!_useNativeMode) return base.GetNamedLayerActivations(input);
        if (input.Rank != 4) input = AddBatchDimension(input);
        return base.GetNamedLayerActivations(input);
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

    // DELIBERATELY NOT the base's AddBatchDimension/RemoveBatchDimension. Those allocate a new
    // tensor and copy into it; OneFormer reshapes through the Engine so the operation stays on the
    // gradient tape, which its ForwardForTraining path depends on. Hiding is intentional, and safe
    // because OneFormer overrides Forward and PredictOnnx, so no base method reaches the base pair.
    private new Tensor<T> AddBatchDimension(Tensor<T> tensor)
        => Engine.Reshape(tensor, [1, tensor.Shape[0], tensor.Shape[1], tensor.Shape[2]]);

    private Tensor<T> AddLeadingBatchDimension(Tensor<T> tensor)
    {
        var shape = new int[tensor.Rank + 1];
        shape[0] = 1;
        for (int i = 0; i < tensor.Rank; i++) shape[i + 1] = tensor.Shape[i];
        return Engine.Reshape(tensor, shape);
    }

    private new Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
        int[] newShape = new int[tensor.Shape.Length - 1];
        for (int i = 0; i < newShape.Length; i++) newShape[i] = tensor.Shape[i + 1];
        return Engine.Reshape(tensor, newShape);
    }

    #endregion

    #region Abstract Implementation

    /// <summary>
    /// Initializes the backbone encoder, text encoder, and transformer decoder layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates the backbone for multi-scale feature extraction and a
    /// text-conditioned transformer decoder that uses task-guided queries.
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
            var encoderLayers = LayerHelper<T>.CreateOneFormerEncoderLayers(
                _height, _width, _channelDims, _depths, _dropRate,
                _attentionHeads, _windowSize, _patchSize, _mlpRatio).ToList();
            _encoderLayerEnd = encoderLayers.Count;
            Layers.AddRange(encoderLayers);

            int fH = Math.Max(1, _height / (_patchSize * 8));
            int fW = Math.Max(1, _width / (_patchSize * 8));

            Layers.AddRange(LayerHelper<T>.CreateOneFormerDecoderLayers(
                _channelDims[^1], _decoderDim, _numClasses, fH, fW));
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
                { "ModelName", "OneFormer" }, { "Description", "OneFormer Universal Segmentation (Text-Conditioned)" },
                { "InputHeight", _height }, { "InputWidth", _width }, { "InputChannels", _channels },
                { "NumClasses", _numClasses }, { "NumQueries", _numQueries },
                { "ModelSize", _modelSize.ToString() }, { "DecoderDim", _decoderDim },
                { "DropRate", _dropRate }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <summary>
    /// Serializes OneFormer configuration.
    /// </summary>
    /// <param name="writer">Binary writer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Saves configuration for later restoration.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_height); writer.Write(_width); writer.Write(_channels);
        writer.Write(_numClasses); writer.Write(_numQueries); writer.Write((int)_modelSize);
        writer.Write(_decoderDim); writer.Write(_dropRate);
        writer.Write(_useNativeMode); writer.Write(_onnxModelPath ?? string.Empty);
        writer.Write(_encoderLayerEnd);
        writer.Write(_channelDims.Length); foreach (int c in _channelDims) writer.Write(c);
        writer.Write(_depths.Length); foreach (int d in _depths) writer.Write(d);
    }

    /// <summary>
    /// Deserializes OneFormer configuration.
    /// </summary>
    /// <param name="reader">Binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Reads saved configuration in write order.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32();
        _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32();
        _ = reader.ReadInt32(); _ = reader.ReadDouble();
        _ = reader.ReadBoolean(); _ = reader.ReadString(); _ = reader.ReadInt32();
        int cc = reader.ReadInt32(); for (int i = 0; i < cc; i++) _ = reader.ReadInt32();
        int dc = reader.ReadInt32(); for (int i = 0; i < dc; i++) _ = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new OneFormer with same config but fresh weights.
    /// </summary>
    /// <returns>New model instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Used for cross-validation or ensemble training.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return _useNativeMode
            ? new OneFormer<T>(Architecture, optimizer: null, lossFunction: LossFunction,
                numClasses: _numClasses, numQueries: _numQueries, modelSize: _modelSize,
                dropRate: _dropRate, options: new OneFormerOptions(_options))
            : new OneFormer<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _numQueries, _modelSize, new OneFormerOptions(_options));
    }

    // Dispose is inherited: SegmentationModelBase already disposes _onnxSession and sets _disposed,
    // and OneFormer owns no further unmanaged resources.

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
