using AiDotNet.LearningRateSchedulers;
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

namespace AiDotNet.ComputerVision.Segmentation.Mamba;

/// <summary>
/// Vision Mamba (Vim): Bidirectional State Space Model for vision.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Efficient image segmentation with linear complexity. Dense prediction replacing ViT backbone.
///
/// Common use cases:
/// - Efficient image segmentation with linear complexity
/// - Dense prediction replacing ViT backbone
/// - Large-scale visual understanding
/// - Memory-efficient visual feature extraction
/// </para>
/// <para>
/// <b>Technical Details:</b> the official segmenter (hustvl/Vim <c>seg/</c>): the Vim backbone without a class token
/// (bidirectional Mamba layers on the patch sequence), the mixer outputs of layers 6, 12, 18 and 24 turned into
/// stride-4, 8, 16 and 32 maps by a BEiT-style neck, mmsegmentation's UPerNet head, and an FCN auxiliary head that adds
/// 0.4 of its cross-entropy to the training loss. Vim-Small adds a 2D rotary position embedding, as its official
/// configuration does. Image sides must be multiples of the 16-pixel patch. The network is checked against the
/// official code.
/// </para>
/// <para>
/// <b>Reference:</b> Zhu et al., "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model", ICML 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Vision Mamba model for efficient semantic segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.MultiClassClassification,
///     inputHeight: 512, inputWidth: 512, inputDepth: 3, outputSize: 150);
/// var model = new VisionMamba&lt;double&gt;(architecture);
/// // Every tunable is now set through the options object; these are the defaults:
/// var custom = new VisionMamba&lt;double&gt;(architecture,
///     options: new VisionMambaOptions { NumClasses = 150, DropRate = 0.1, ModelSize = VisionMambaModelSize.Tiny });
///
/// // Or load a pre-trained ONNX model for linear-complexity segmentation
/// var onnxModel = new VisionMamba&lt;double&gt;(architecture, "vim.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model", "https://arxiv.org/abs/2401.09417", Year = 2024, Authors = "Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, Xinggang Wang")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 6e-5, WeightDecay = 0.01, ReferenceBatchSize = 16,
                Schedule = LearningRateSchedulerType.LinearWarmup, WarmupFraction = 0.009375,
                PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear, MinLearningRate = 0,
                Source = "Zhu et al. 2024, Sec. 4.2: AdamW with a weight decay of 0.01 and a total batch size of 16, an initial learning rate of 6e-5 with linear decay, a linear warmup of 1,500 iterations, and 160K iterations of training (1,500 / 160,000 of the run).")]
[TaskVariantOf(typeof(AiDotNet.NeuralNetworks.VisionMambaModel<>), "semantic segmentation")]
public partial class VisionMamba<T> : Common.SemanticSegmentationBase<T>
{
    private readonly VisionMambaOptions _options;
    private VimSegmentationNetwork<T>? _network;

    /// <summary>Draws stochastic-depth and dropout decisions during training.</summary>
    private Random _trainingRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(43);

    internal bool UseNativeMode => _useNativeMode;
    internal VisionMambaModelSize ModelSize => _options.ModelSize;

    /// <summary>The network (backbone, neck, decode and auxiliary heads).</summary>
    internal VimSegmentationNetwork<T> Network => _network ?? throw new InvalidOperationException("The native network is not initialized.");

    /// <summary>Initializes VisionMamba in native (trainable) mode.</summary>
    /// <param name="architecture">Neural network architecture defining input dimensions (multiples of 16 pixels).</param>
    /// <param name="optimizer">Gradient-based optimizer (default: the paper's segmentation recipe).</param>
    /// <param name="lossFunction">Loss function reported by the base; training minimizes the decode and auxiliary
    /// cross-entropies.</param>
    /// <param name="options">Optional model options.</param>
    public VisionMamba(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        VisionMambaOptions? options = null)
        : base(architecture, optimizer, lossFunction, (options ??= new VisionMambaOptions()).NumClasses)
    {
        _options = options; Options = _options;
        ApplyVisionMambaInputFallback(architecture);
        InitializeLayers();
    }

    /// <summary>Initializes VisionMamba in ONNX (inference-only) mode.</summary>
    public VisionMamba(NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        VisionMambaOptions? options = null)
        : base(architecture, onnxModelPath, (options ??= new VisionMambaOptions()).NumClasses)
    {
        _options = options; Options = _options;
        ApplyVisionMambaInputFallback(architecture);
        InitializeLayers();
    }

    /// <summary>VisionMamba's fallback geometry is 224x224 when the architecture carries none.</summary>
    private void ApplyVisionMambaInputFallback(NeuralNetworkArchitecture<T> architecture)
    {
        if (architecture.InputHeight <= 0) _height = 224;
        if (architecture.InputWidth <= 0) _width = 224;
    }

    /// <summary>The official configuration of each size: Vim-Ti (width 192) and Vim-S (width 384, with RoPE), 24 layers,
    /// features from layers 6/12/18/24, a decode head as wide as the backbone and a 256-channel auxiliary head.</summary>
    private VimSegmentationConfiguration Configuration()
    {
        bool small = _options.ModelSize == VisionMambaModelSize.Small;
        int dim = small ? 384 : 192;
        return new VimSegmentationConfiguration
        {
            Backbone = new VimConfiguration
            {
                ImageHeight = Canvas(_height), ImageWidth = Canvas(_width), Channels = _channels, PatchSize = 16, Stride = 16, Dim = dim,
                Depth = 24, StateDim = 16, DropPathRate = _options.DropRate, ClassToken = false, Rope = small, RopeResidual = small,
            },
            OutIndices = new[] { 5, 11, 17, 23 }, Classes = _numClasses, HeadChannels = dim, AuxiliaryChannels = 256,
        };
    }

    protected override void InitializeLayers()
    {
        if (_options is null || _network is not null) return;
        if (!_useNativeMode) { ClearLayers(); return; }
        if (Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException("VisionMamba builds the paper's network; custom layers are not supported.");
        var layers = new List<LayerBase<T>>();
        _network = new VimSegmentationNetwork<T>(Engine, layers, Configuration());
        int seed = _options.Seed ?? 42;
        _network.InitializeLikeReference(Tensors.Helpers.RandomHelper.CreateSeededRandom(seed));
        _trainingRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(seed + 1);
        Layers.AddRange(layers);
        _encoderLayerEnd = layers.Count;
    }

    /// <summary>Loads the official model's state dict (<c>backbone.*</c>, <c>decode_head.*</c>, <c>auxiliary_head.*</c>).</summary>
    internal void LoadTorchWeights(Func<string, int[], double[]> read) => Network.LoadTorchWeights(read);

    /// <summary>The class logits <c>[B, classes, H, W]</c> (or <c>[classes, H, W]</c>) at the image's resolution.</summary>
    /// <remarks>An image whose sides are not multiples of the 16-pixel patch is zero-padded on the bottom and right to the
    /// network's canvas and its logits cropped back, as mmseg pads to the size divisor (pad value 0) and crops.</remarks>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4; if (!hasBatch) input = AddBatchDimension(input);
        int h = input.Shape[2], w = input.Shape[3];
        var logits = Crop(Network.Forward(PadToCanvas(input), IsTrainingMode, _trainingRandom, withAuxiliary: false).Logits, h, w);
        return hasBatch ? logits : RemoveBatchDimension(logits);
    }

    /// <summary>A side rounded up to the 16-pixel patch.</summary>
    private static int Canvas(int side) => (side + 15) / 16 * 16;

    /// <summary>Zero-pads <c>[B, C, h, w]</c> on the bottom and right to the patch multiple.</summary>
    private Tensor<T> PadToCanvas(Tensor<T> image)
    {
        int b = image.Shape[0], c = image.Shape[1], h = image.Shape[2], w = image.Shape[3];
        int height = Canvas(h), width = Canvas(w);
        if (width > w) image = Engine.TensorConcatenate(new[] { image, new Tensor<T>(new[] { b, c, h, width - w }) }, 3);
        if (height > h) image = Engine.TensorConcatenate(new[] { image, new Tensor<T>(new[] { b, c, height - h, width }) }, 2);
        return image;
    }

    /// <summary>The top-left <c>h x w</c> of <c>[B, C, H, W]</c> logits.</summary>
    private Tensor<T> Crop(Tensor<T> logits, int h, int w)
        => logits.Shape[2] == h && logits.Shape[3] == w
            ? logits
            : Engine.TensorSlice(logits, new[] { 0, 0, 0, 0 }, new[] { logits.Shape[0], logits.Shape[1], h, w });

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => Forward(input);

    /// <summary>
    /// One step on the decode head's pixel cross-entropy plus 0.4 of the auxiliary head's (mmseg's loss weights),
    /// against a target <c>[B, classes, H, W]</c> of one-hot (or soft) class maps.
    /// </summary>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode) throw new InvalidOperationException("Training is not supported in ONNX mode. Use the native mode constructor for training.");
        if (input.Shape.Length == 3) input = input.Reshape(new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2] });
        else if (input.Shape.Length != 4) throw new ArgumentException($"Training requires rank 3 [C,H,W] or 4 [B,C,H,W], got rank {input.Shape.Length}.", nameof(input));
        if (expectedOutput.Shape.Length == 3) expectedOutput = expectedOutput.Reshape(new[] { 1, expectedOutput.Shape[0], expectedOutput.Shape[1], expectedOutput.Shape[2] });
        else if (expectedOutput.Shape.Length != 4) throw new ArgumentException($"Expected output requires rank 3 [C,H,W] or 4 [B,C,H,W], got rank {expectedOutput.Shape.Length}.", nameof(expectedOutput));
        if (input.Shape[0] != expectedOutput.Shape[0]) throw new ArgumentException($"Batch size mismatch: input has {input.Shape[0]} but expectedOutput has {expectedOutput.Shape[0]}.", nameof(expectedOutput));
        var network = Network;
        var target = expectedOutput;
        int height = input.Shape[2], width = input.Shape[3];
        var image = PadToCanvas(input);
        SetTrainingMode(true);
        try
        {
            TrainWithCustomLoss(image, _ =>
            {
                var (logits, auxiliary) = network.Forward(image, training: true, _trainingRandom, withAuxiliary: true);
                return Engine.TensorAdd(PixelCrossEntropy(Crop(logits, height, width), target),
                    Engine.TensorMultiplyScalar(PixelCrossEntropy(Crop(auxiliary!, height, width), target),
                        NumOps.FromDouble(network.Configuration.AuxiliaryWeight)));
            }, Optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
        return true;
    }

    /// <summary>The mean over pixels of −Σ_c t_c·log softmax(x)_c, the class axis being axis 1.</summary>
    private Tensor<T> PixelCrossEntropy(Tensor<T> logits, Tensor<T> target)
    {
        var shape = logits.Shape.ToArray();
        // Subtracting a per-pixel constant leaves the softmax unchanged; the maximum is taken without a gradient.
        var maximum = new Tensor<T>(new[] { shape[0], 1, shape[2], shape[3] });
        for (int b = 0; b < shape[0]; b++)
            for (int p = 0; p < shape[2] * shape[3]; p++)
            {
                double m = double.NegativeInfinity;
                for (int c = 0; c < shape[1]; c++) m = Math.Max(m, NumOps.ToDouble(logits[(b * shape[1] + c) * shape[2] * shape[3] + p]));
                maximum[b * shape[2] * shape[3] + p] = NumOps.FromDouble(m);
            }
        var centered = Engine.TensorSubtract(logits, Engine.TensorBroadcastTo(maximum, shape));
        var logSumExp = Engine.TensorLog(Engine.ReduceSum(Engine.TensorExp(centered), new[] { 1 }, false));
        var picked = Engine.ReduceSum(Engine.TensorMultiply(centered, target), new[] { 1 }, false);
        var mass = Engine.ReduceSum(target, new[] { 1 }, false);
        return Engine.ReduceMean(Engine.TensorSubtract(Engine.TensorMultiply(logSumExp, mass), picked), new[] { 0, 1, 2 }, false);
    }

    /// <inheritdoc />
    /// <remarks>Every layer is sized at construction and <see cref="NeuralNetworkBase{T}.Layers"/> is not a chain.</remarks>
    protected override void ResolveLazyLayerShapes()
    {
    }

    /// <inheritdoc />
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input) => ObservedLayerActivations(input);

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

    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "VisionMamba" }, { "InputHeight", _height }, { "InputWidth", _width }, { "NumClasses", _numClasses }, { "ModelSize", _options.ModelSize.ToString() }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
        ModelDataProvider = () => SerializeForMetadata()
    };
}
