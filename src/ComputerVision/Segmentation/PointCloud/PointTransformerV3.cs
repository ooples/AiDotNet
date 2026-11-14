using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// Point Transformer V3: Simpler, Faster, Stronger -- per-point semantic segmentation of 3D point clouds.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Give the model a point cloud -- a list of points, each with its xyz position
/// and a few extra values such as colour and surface normal -- and it returns a score for every class
/// at every point. It is used for LiDAR and indoor-scan understanding.
/// </para>
/// <para>
/// <b>Architecture</b> (Wu et al., CVPR 2024; Pointcept <c>PT-v3m1</c>), built by
/// <see cref="PointTransformerV3Encoder{T}"/>:
/// </para>
/// <list type="bullet">
/// <item>Serialization: points are voxelised (<c>grid_size</c> 0.02 m) and ordered along Z-order and
/// Hilbert curves and their xy-transposed variants; each block attends in one order, cycling through
/// them, and training shuffles the orders after every pooling.</item>
/// <item>Serialized patch attention: multi-head attention among patches of 1024 consecutive points of
/// the current order, with xCPE (a 3x3x3 submanifold convolution) before it.</item>
/// <item>Grid pooling between stages (stride 2, max-pool of the projected features) and unpooling that
/// adds the coarse features back to their points, U-Net style.</item>
/// <item>Encoder depths (2, 2, 2, 6, 2), channels (32, 64, 128, 256, 512), heads (2, 4, 8, 16, 32);
/// decoder depths (2, 2, 2, 2), channels (64, 64, 128, 256); pre-norm, MLP ratio 4, drop path 0.3;
/// a linear segmentation head on the finest decoder features.</item>
/// </list>
/// <para>
/// <b>Input</b>: <c>[points, 3 + InputChannels]</c> or <c>[batch, points, 3 + InputChannels]</c>,
/// xyz in metres first. <b>Output</b>: class logits <c>[..., points, NumClasses]</c>.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     InputType.TwoDimensional, NeuralNetworkTaskType.ImageSegmentation,
///     inputSize: 9, outputSize: 20);
/// var options = new PointTransformerV3Options { NumClasses = 20 };
/// var model = new PointTransformerV3&lt;float&gt;(architecture, options: options);
///
/// var cloud = Tensor&lt;float&gt;.CreateRandom(4096, 9);   // xyz, colour, normal
/// Tensor&lt;float&gt; logits = model.Predict(cloud);          // [4096, 20]
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.ThreeD)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Point Transformer V3: Simpler, Faster, Stronger", "https://arxiv.org/abs/2312.10035", Year = 2024,
    Authors = "Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, Hengshuang Zhao")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Length, TensorAxis.Features, Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Length, TensorAxis.Classes, Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public partial class PointTransformerV3<T> : Common.SemanticSegmentationBase<T>
{
    private readonly PointTransformerV3Options _options;

    /// <summary>
    /// Initializes PointTransformerV3 in native (trainable) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture; custom layers in it replace the default network.</param>
    /// <param name="optimizer">Gradient-based optimizer (default: AdamW).</param>
    /// <param name="lossFunction">Loss function (default: per-point cross-entropy over the last axis).</param>
    /// <param name="options">Model options; the defaults are the PTv3 m1 base configuration.</param>
    public PointTransformerV3(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        PointTransformerV3Options? options = null)
        : base(architecture, optimizer, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(-1),
            (options ??= new PointTransformerV3Options()).NumClasses)
    {
        _options = options;
        Options = _options;
        _options.Validate();
        _channels = _options.InputChannels + 3;
        InitializeLayers();
    }

    /// <summary>
    /// Initializes PointTransformerV3 in ONNX (inference-only) mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="options">Optional model options.</param>
    /// <exception cref="ArgumentException">Thrown if the ONNX model path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    /// <exception cref="InvalidOperationException">Thrown if the ONNX runtime fails to load the model.</exception>
    public PointTransformerV3(NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        PointTransformerV3Options? options = null)
        : base(architecture, onnxModelPath, (options ??= new PointTransformerV3Options()).NumClasses)
    {
        _options = options;
        Options = _options;
        _options.Validate();
        _channels = _options.InputChannels + 3;
        InitializeLayers();
    }

    internal bool UseNativeMode => _useNativeMode;

    /// <summary>The backbone, or null in ONNX mode or with custom architecture layers.</summary>
    internal PointTransformerV3Encoder<T>? Encoder { get; private set; }

    /// <summary>The data-dependent serialization and pooling indices cannot be fused into a compiled plan.</summary>
    protected override bool SupportsFusedCompiledTraining => false;

    /// <inheritdoc />
    public override IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => PointCloudContract(inputRank, _numClasses);

    /// <summary>Argmax over the class (last) axis: one class index per point.</summary>
    public override Tensor<T> GetClassMap(Tensor<T> image) => LastAxisArgMax(Segment(image));

    /// <summary>Softmax over the class (last) axis.</summary>
    public override Tensor<T> GetProbabilityMap(Tensor<T> image) => Engine.TensorSoftmax(Segment(image), -1);

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input) => _useNativeMode ? Forward(input) : PredictOnnx(input);

    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        var features = input;
        foreach (var layer in Layers) features = layer.Forward(features);
        return features;
    }

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
        {
            ClearLayers();
            return;
        }

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = Math.Max(0, Architecture.Layers.Count - 1);
            return;
        }

        var o = _options;
        Encoder = new PointTransformerV3Encoder<T>(o.InputChannels, false, o.GridSize, o.Strides, o.EncoderDepths,
            o.EncoderChannels, o.EncoderNumHeads, o.EncoderPatchSizes, o.DecoderDepths, o.DecoderChannels,
            o.DecoderNumHeads, o.DecoderPatchSizes, o.MlpRatio, o.DropPathRate, o.EnableConditionalPositionalEncoding,
            o.PreNorm, o.ShuffleOrders, o.SerializationOrders, PointTransformerV3Revision.M1Base, false, false, o.Seed);
        Layers.Add(Encoder);
        _encoderLayerEnd = 1;
        // Pointcept seg_head: Linear(dec_channels[0], num_classes).
        Layers.Add(new DenseLayer<T>(_numClasses, (IActivationFunction<T>?)null));
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "PointTransformerV3" },
            { "Revision", PointTransformerV3Revision.M1Base.ToString() },
            { "InputChannels", _options.InputChannels },
            { "NumClasses", _numClasses },
            { "EncoderDepths", string.Join(",", _options.EncoderDepths) },
            { "EncoderChannels", string.Join(",", _options.EncoderChannels) },
            { "UseNativeMode", _useNativeMode },
            { "NumLayers", Layers.Count }
        },
        ModelDataProvider = () => SerializeForMetadata()
    };

    /// <summary>The per-point output contract shared by PTv3 and Sonata.</summary>
    internal static IReadOnlyList<OutputAxisContract>? PointCloudContract(int inputRank, int classes)
    {
        var points = new OutputAxisContract(TensorAxis.Length, AxisRelation.Same(TensorAxis.Length));
        var logits = new OutputAxisContract(TensorAxis.Classes, AxisRelation.Fixed(classes));
        return inputRank switch
        {
            2 => new[] { points, logits },
            3 => new[] { new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)), points, logits },
            _ => null
        };
    }

    /// <summary>Argmax over the last axis, returned in <typeparamref name="T"/>.</summary>
    internal static Tensor<T> LastAxisArgMax(Tensor<T> logits)
    {
        var indices = AiDotNetEngine.Current.TensorArgMax(logits, logits.Rank - 1).ToArray();
        var asFloat = new float[indices.Length];
        for (int i = 0; i < indices.Length; i++) asFloat[i] = indices[i];
        var values = new T[indices.Length];
        MathHelper.GetNumericOperations<T>().FromFloatSpan(asFloat, values);
        var shape = new int[logits.Rank - 1];
        for (int i = 0; i < shape.Length; i++) shape[i] = logits.Shape[i];
        return new Tensor<T>(values, shape);
    }
}
