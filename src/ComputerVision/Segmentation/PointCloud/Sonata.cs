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
/// Sonata: self-supervised learning of reliable point representations (point self-distillation on a
/// Point Transformer V3 encoder).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Sonata is a way to pretrain a 3D point cloud network without labels so that its point
/// features are useful on their own; a simple linear layer on top of them already segments a scene well.
/// </para>
/// <para>
/// <b>Reference:</b> Wu et al., "Sonata: Self-Supervised Learning of Reliable Point Representations",
/// CVPR 2025 (arXiv:2503.16429).
/// </para>
/// <para>
/// <b>Migration in progress: this class does not yet implement that paper.</b> Its attribution was
/// previously wrong twice over: a reference to a paper that does not exist ("Sonata and Concerto: Mamba for
/// 3D Point Clouds") and a <c>[ResearchPaper]</c> attribute pointing at Mamba3D (arXiv:2404.14966), an
/// unrelated model. The citation now names the real Sonata.
/// </para>
/// <para>
/// What the published method is, and what still has to be built here:
/// </para>
/// <list type="bullet">
/// <item><b>Backbone.</b> A Point Transformer V3 encoder (serialized patch attention over space-filling-curve
/// orderings, grid pooling, LayerNorm instead of BatchNorm), depths [3, 3, 3, 12, 3] and widths
/// [48, 96, 192, 384, 512]. The current backbone is a strided 2D convolution stack with BatchNorm over an
/// image-shaped grid, not PTv3.</item>
/// <item><b>Pretraining.</b> Point self-distillation without a decoder: an EMA teacher (momentum 0.994 to 1),
/// 2 global, 4 local and 2 masked views, online clustering heads with Sinkhorn-Knopp centering and KoLeo
/// regularization, and encoder features up-cast two levels before the loss. None of this runs here yet.</item>
/// <item><b>Downstream.</b> Linear probe, decoder probe or full fine-tuning on the pretrained encoder. The
/// current class trains its 3-convolution segmentation head end to end instead.</item>
/// </list>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     InputType.ThreeDimensional, NeuralNetworkTaskType.ImageSegmentation,
///     inputSize: 6, outputSize: 40);
///
/// var trainPoints = Tensor&lt;float&gt;.CreateRandom(4, 128, 6);
/// var trainLabels = Tensor&lt;float&gt;.CreateRandom(4, 128, 40);
///
/// // Every tunable is set through the options object; these are the defaults.
/// var options = new SonataOptions { NumClasses = 40, DropRate = 0.1, ModelSize = SonataModelSize.Base };
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new Sonata&lt;float&gt;(architecture, options: options))
///     .Build(trainPoints, trainLabels);
///
/// var pointCloudTensor = Tensor&lt;float&gt;.CreateRandom(1, 128, 6);
/// Tensor&lt;float&gt; segmentation = result.Predict(pointCloudTensor);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.ThreeD)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Sonata: Self-Supervised Learning of Reliable Point Representations", "https://arxiv.org/abs/2503.16429", Year = 2025, Authors = "Xiaoyang Wu, Daniel DeTone, Duncan Frost, Tianwei Shen, Chris Xie, Nan Yang, Jakob Engel, Richard Newcombe, Hengshuang Zhao, Julian Straub")]
public partial class Sonata<T> : Common.SemanticSegmentationBase<T>
{
    private readonly SonataOptions _options;

    #region Fields
    // Only Sonata's OWN configuration lives here. _height, _width, _channels, _numClasses,
    // _useNativeMode, _onnxModelPath, _onnxSession, _optimizer, _disposed and _encoderLayerEnd all
    // come from SemanticSegmentationBase -> SegmentationModelBase.
    private readonly SonataModelSize _modelSize;
    private readonly int[] _channelDims;
    private readonly int _decoderDim;
    private readonly int[] _depths;
    private readonly double _dropRate;
    #endregion

    #region Properties
    // SupportsTraining, NumClasses, InputHeight, InputWidth and IsOnnxMode are all inherited and
    // say exactly the same thing.
    internal bool UseNativeMode => _useNativeMode;
    internal SonataModelSize ModelSize => _modelSize;
    #endregion

    #region Constructors
    /// <summary>
    /// Initializes Sonata in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: CrossEntropyLoss).</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a trainable Sonata model.
    /// </para>
    /// </remarks>
    public Sonata(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        SonataOptions? options = null)
        // The base resolves height/width/channels/numClasses/native-mode from the architecture and
        // defaults `optimizer` lazily via CreateDefaultOptimizer(), so null is passed straight through.
        : base(architecture, optimizer, lossFunction, (options ??= new SonataOptions()).NumClasses)
    {
        _options = options; Options = _options;
        // Point clouds are [C, N] not images: this model's own fallback geometry is 1x1x6, not the
        // base's 512x512x3.
        if (architecture.InputHeight <= 0) _height = 1;
        if (architecture.InputWidth <= 0) _width = 1;
        if (architecture.InputDepth <= 0) _channels = 6;
        _modelSize = _options.ModelSize; _dropRate = _options.DropRate;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(_options.ModelSize);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes Sonata in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture defining input dimensions.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="options">Optional model options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads a pre-trained Sonata from ONNX for inference.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public Sonata(NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        SonataOptions? options = null)
        // The base validates the path, sets ONNX mode, resolves the input geometry and opens the
        // InferenceSession.
        : base(architecture, onnxModelPath, (options ??= new SonataOptions()).NumClasses)
    {
        _options = options; Options = _options;
        // Point clouds are [C, N] not images: this model's own fallback geometry is 1x1x6, not the
        // base's 512x512x3.
        if (architecture.InputHeight <= 0) _height = 1;
        if (architecture.InputWidth <= 0) _width = 1;
        if (architecture.InputDepth <= 0) _channels = 6;
        _modelSize = _options.ModelSize; _dropRate = _options.DropRate;
        (_channelDims, _depths, _decoderDim) = GetModelConfig(_options.ModelSize);
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

    #endregion

    #region Private Methods
    private static (int[] ChannelDims, int[] Depths, int DecoderDim) GetModelConfig(SonataModelSize modelSize) => modelSize switch
    {
        SonataModelSize.Base => ([48, 96, 192, 384], [2, 2, 6, 2], 256),
        SonataModelSize.Large => ([64, 128, 256, 512], [2, 4, 8, 4], 256),
        _ => ([48, 96, 192, 384], [2, 2, 6, 2], 256)
    };

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
        NumOps.ToFloatSpan(input.Data.Span.Slice(0, input.Length), inputData);
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "images";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };
        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var outputData = new T[outputTensor.Length];
        NumOps.FromFloatSpan(outputTensor.ToDenseTensor().Buffer.Span, outputData);
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
            var encoderLayers = LayerHelper<T>.CreateSonataEncoderLayers(_channels, _height, _width, _channelDims, _depths, _dropRate).ToList();
            _encoderLayerEnd = encoderLayers.Count; Layers.AddRange(encoderLayers);
            int fH = _height / 32, fW = _width / 32;
            var decoderLayers = LayerHelper<T>.CreateSonataDecoderLayers(_channelDims[^1], _decoderDim, _numClasses, fH, fW);
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
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "Sonata" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "ModelSize", _modelSize.ToString() }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
        ModelDataProvider = () => SerializeForMetadata()
    };
    #endregion

    // NumClasses, InputHeight, InputWidth, IsOnnxMode and Segment come from SegmentationModelBase;
    // GetClassMap (argmax of Segment) and GetProbabilityMap (softmax of Segment) come from
    // SemanticSegmentationBase and are the same two expressions this file used to spell out.
}
