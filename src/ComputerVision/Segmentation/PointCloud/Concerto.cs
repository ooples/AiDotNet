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

namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// Concerto: Hybrid Mamba-Transformer backbone for 3D point clouds.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> High-accuracy 3D point cloud segmentation. Multi-modal 3D scene understanding.
///
/// Common use cases:
/// - High-accuracy 3D point cloud segmentation
/// - Multi-modal 3D scene understanding
/// - Dense 3D prediction tasks
/// - Advanced LiDAR perception
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Hybrid Mamba + Transformer blocks for balanced efficiency and accuracy
/// - Global Mamba branches + local Transformer attention
/// - Better quality than pure Mamba while remaining efficient
/// - Serialized point processing with space-filling curves
/// </para>
/// <para>
/// <b>Reference:</b> "Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial
/// Representations" (arXiv:2510.23607).
/// </para>
/// <para>
/// <b>Migration in progress — this class does not yet implement that paper.</b> Its attribution
/// was previously wrong in three separate ways: a fabricated reference ("Sonata and Concerto:
/// Mamba for 3D Point Clouds", which does not exist), a <c>[ResearchPaper]</c> attribute pointing
/// at Mamba3D (arXiv:2404.14966 — a real but unrelated model by different authors), and an
/// implementation matching neither. The citation now names the real Concerto, and
/// <see cref="ConcertoOptions"/> carries that paper's hyperparameters.
/// </para>
/// <para>
/// What the published method actually is, and what still has to be built here:
/// </para>
/// <list type="bullet">
/// <item><b>3D intra-modal self-distillation.</b> A Point Transformer V3 student is trained to
/// match a momentum-updated (EMA) teacher under a DINOv2-style online-clustering cross-entropy
/// objective at decoder upcast level 2. Neither the teacher/student pair nor the clustering loss
/// exists here yet.</item>
/// <item><b>2D-3D cross-modal joint embedding.</b> A frozen DINOv2-L encoder at 518x518 supplies
/// image patch features; 3D points are projected into the image and depth-checked for visibility
/// (|d_c - d_proj| &lt; 0.01 m), point features falling in a patch are mean-pooled, and the result
/// is matched to the image patch embedding by cosine similarity at upcast level 3. This requires
/// paired imagery plus camera intrinsics and extrinsics, which this type's current API does not
/// accept.</item>
/// <item><b>Objective.</b> The two terms are combined at the paper's 2:2 weight ratio.</item>
/// </list>
/// <para>
/// Because Concerto is a PRETRAINING method, segmentation is a downstream probe on the learned
/// representation rather than the training objective. <see cref="ISemanticSegmentation{T}"/> is
/// retained so existing consumers keep working, but the current backbone is the previous
/// unattributed hybrid Mamba/Transformer stack, not PTv3, and no self-supervised objective runs.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Concerto hybrid Mamba-Transformer model for 3D point cloud segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputSize: 6, outputSize: 40, networkType: NetworkType.Classification);
/// var concerto = new Concerto&lt;float&gt;(architecture,
///     numClasses: 40, modelSize: ConcertoModelSize.Base, dropRate: 0.1);
/// Tensor&lt;float&gt; segmentationMap = concerto.Forward(pointCloudTensor);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.ThreeD)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations", "https://arxiv.org/abs/2510.23607", Year = 2025)]
public partial class Concerto<T> : Common.SemanticSegmentationBase<T>
{
    private readonly ConcertoOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    // Only Concerto's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from SemanticSegmentationBase -> SegmentationModelBase.
    private readonly ConcertoModelSize _modelSize;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    #endregion

    #region Properties
    // SupportsTraining, NumClasses, InputHeight, InputWidth and IsOnnxMode are all inherited and
    // say exactly the same thing.
    internal bool UseNativeMode => _useNativeMode;
    internal ConcertoModelSize ModelSize => _modelSize;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes Concerto in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="numClasses">Number of segmentation classes (default: 40).</param>
    /// <param name="modelSize">Model size variant (default: Base).</param>
    /// <param name="dropRate">Dropout rate (default: 0.1).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable Concerto model.
    /// </para>
    /// </remarks>
    public Concerto(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 40,
        ConcertoModelSize modelSize = ConcertoModelSize.Base, double dropRate = 0.1,
        ConcertoOptions? options = null)
        // The base resolves height/width/channels/numClasses/native-mode from the architecture and
        // defaults `optimizer` lazily via CreateDefaultOptimizer(), so null is passed straight through.
        : base(architecture, optimizer, lossFunction, numClasses)
    {
        _options = options ?? new ConcertoOptions(); Options = _options;
        // Point clouds are [C, N] not images: this model's own fallback geometry is 1x1x6, not the
        // base's 512x512x3.
        if (architecture.InputHeight <= 0) _height = 1;
        if (architecture.InputWidth <= 0) _width = 1;
        if (architecture.InputDepth <= 0) _channels = 6;
        _modelSize = modelSize; _dropRate = dropRate;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(modelSize);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes Concerto in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="numClasses">Number of segmentation classes (default: 40).</param>
    /// <param name="modelSize">Model size for metadata (default: Base).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained Concerto from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public Concerto(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 40, ConcertoModelSize modelSize = ConcertoModelSize.Base,
        ConcertoOptions? options = null)
        // The base validates the path, sets ONNX mode, resolves the input geometry and opens the
        // InferenceSession.
        : base(architecture, onnxModelPath, numClasses)
    {
        _options = options ?? new ConcertoOptions(); Options = _options;
        // Point clouds are [C, N] not images: this model's own fallback geometry is 1x1x6, not the
        // base's 512x512x3.
        if (architecture.InputHeight <= 0) _height = 1;
        if (architecture.InputWidth <= 0) _width = 1;
        if (architecture.InputDepth <= 0) _channels = 6;
        _modelSize = modelSize; _dropRate = 0.1;
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
    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(ConcertoModelSize modelSize) => modelSize switch
    {
        ConcertoModelSize.Base => ([48, 96, 192, 384], [2, 2, 6, 2], 256),
        ConcertoModelSize.Large => ([64, 128, 256, 512], [2, 4, 8, 4], 256),
        _ => ([48, 96, 192, 384], [2, 2, 6, 2], 256)
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
            var encoderLayers = LayerHelper<T>.CreateConcertoEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateConcertoDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "Concerto" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "ModelSize", _modelSize.ToString() }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
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
        ? new Concerto<T>(Architecture, _optimizer, LossFunction, _numClasses, _modelSize, _dropRate, _options)
        : new Concerto<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _modelSize, _options);
    #endregion

    // NumClasses, InputHeight, InputWidth, IsOnnxMode and Segment come from SegmentationModelBase;
    // GetClassMap (argmax of Segment) and GetProbabilityMap (softmax of Segment) come from
    // SemanticSegmentationBase and are the same two expressions this file used to spell out.
}
