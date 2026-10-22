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

namespace AiDotNet.ComputerVision.Segmentation.Semantic;

/// <summary>
/// SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SegNeXt is a semantic segmentation model that labels every pixel
/// in an image with a category (e.g., road, tree, building). Unlike transformer-based models,
/// SegNeXt uses a purely convolutional backbone called MSCAN (Multi-Scale Convolutional Attention
/// Network) that achieves better accuracy than many transformers while being simpler and faster.
///
/// Common use cases:
/// - Autonomous driving (road, lane, sidewalk parsing)
/// - Drone/satellite imagery analysis
/// - Indoor scene understanding
/// - Real-time segmentation on resource-constrained devices
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - MSCAN encoder: a convolutional stem, then blocks of multi-scale convolutional attention (a 5x5 depth-wise
///   convolution plus 1x7/7x1, 1x11/11x1 and 1x21/21x1 strip branches, mixed by a 1x1 convolution and multiplied
///   into the input) and depth-wise convolutional MLPs, with BatchNorm, layer scale and stochastic depth
/// - LightHamHead decoder: stages 2-4 at 1/8 scale, concatenated and factored by a non-negative matrix
///   factorization (the Hamburger) for global context, then a 1x1 classifier; logits are upsampled to the image
/// - Matches mmsegmentation's MSCAN and LightHamHead to 1e-9 (SegNeXtReferenceTests) and loads their checkpoints
/// - Trains with the paper's recipe: AdamW 6e-5 (head x10, no decay on norms), warm-up then poly decay
/// </para>
/// <para>
/// <b>Reference:</b> Guo et al., "SegNeXt: Rethinking Convolutional Attention Design for
/// Semantic Segmentation", NeurIPS 2022.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a SegNeXt model for convolutional semantic segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 512, inputWidth: 512, inputDepth: 3, outputSize: 150);
/// var model = new SegNeXt&lt;double&gt;(architecture);
/// // Every tunable is now set through the options object; these are the defaults:
/// var custom = new SegNeXt&lt;double&gt;(architecture,
///     options: new SegNeXtOptions { NumClasses = 150, DropRate = 0.1, ModelSize = SegNeXtModelSize.Tiny });
///
/// // Or load a pre-trained ONNX model for inference
/// var onnxModel = new SegNeXt&lt;double&gt;(architecture, "segnext.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation", "https://arxiv.org/abs/2209.08575", Year = 2022, Authors = "Guo et al.")]
public partial class SegNeXt<T> : Common.SemanticSegmentationBase<T>
{
    private readonly SegNeXtOptions _options;
    private SegNeXtNetwork<T>? _network;
    private Random _trainingRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(43);

    // The NMF's bases [1, C, R], drawn once (the official code's rand_init=False) and saved with the model; null
    // draws fresh bases on every forward (rand_init=True), or in ONNX mode.
    [Buffer(Availability = AiDotNet.Models.Parameters.ParameterAvailability.Conditional)]
    private Tensor<T>? _nmfBases;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    internal bool UseNativeMode => _useNativeMode;
    internal SegNeXtModelSize ModelSize => _options.ModelSize;
    internal SegNeXtNetwork<T> Network => _network ?? throw new InvalidOperationException("The native network is not initialized.");

    /// <summary>
    /// Creates a trainable SegNeXt with the paper's architecture for <see cref="SegNeXtOptions.ModelSize"/>.
    /// </summary>
    /// <param name="architecture">The input image's shape [C, H, W]; any height and width work.</param>
    /// <param name="optimizer">An optimizer to use instead of the paper's recipe.</param>
    /// <param name="lossFunction">Kept for the segmentation family's signature; SegNeXt trains on pixel cross-entropy.</param>
    /// <param name="options">The size, the class count and the training recipe.</param>
    public SegNeXt(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        SegNeXtOptions? options = null)
        : base(architecture, optimizer, lossFunction, (options ??= new SegNeXtOptions()).NumClasses)
    {
        _options = options;
        Options = _options;
        InitializeLayers();
    }

    /// <summary>Loads an ONNX export for inference.</summary>
    /// <param name="architecture">The ONNX model's input shape.</param>
    /// <param name="onnxModelPath">The ONNX file.</param>
    /// <param name="options">The class count (the export fixes the rest).</param>
    public SegNeXt(NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        SegNeXtOptions? options = null)
        : base(architecture, onnxModelPath, (options ??= new SegNeXtOptions()).NumClasses)
    {
        _options = options;
        Options = _options;
        InitializeLayers();
    }

    // mmsegmentation's configs: MSCAN-T/S/B/L and the LightHamHead width.
    private SegNeXtConfiguration Configuration()
    {
        var (dims, depths, head, dropPath) = _options.ModelSize switch
        {
            SegNeXtModelSize.Small => (new[] { 64, 128, 320, 512 }, new[] { 2, 2, 4, 2 }, 256, 0.1),
            SegNeXtModelSize.Base => (new[] { 64, 128, 320, 512 }, new[] { 3, 3, 12, 3 }, 512, 0.1),
            SegNeXtModelSize.Large => (new[] { 64, 128, 320, 512 }, new[] { 3, 5, 27, 3 }, 1024, 0.3),
            _ => (new[] { 32, 64, 160, 256 }, new[] { 3, 3, 5, 2 }, 256, 0.1),
        };
        int channels = _options.HeadChannels ?? head;
        return new SegNeXtConfiguration
        {
            InputChannels = _channels,
            EmbedDims = _options.EmbedDims ?? dims,
            Depths = _options.Depths ?? depths,
            MlpRatios = new[] { 8, 8, 4, 4 },
            DropPathRate = _options.DropPathRate ?? dropPath,
            HeadChannels = channels,
            HamChannels = channels,
            Classes = _numClasses,
            MatrixRank = _options.MatrixRank,
            HeadDropout = _options.DropRate,
        };
    }

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (_options is null || _network is not null) return;
        if (!_useNativeMode)
        {
            ClearLayers();
            return;
        }
        if (Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException("SegNeXt builds the paper's network; custom layers are not supported.");
        var layers = new List<LayerBase<T>>();
        _network = new SegNeXtNetwork<T>(Engine, layers, Configuration());
        int seed = _options.Seed ?? 42;
        _network.InitializeLikeReference(Tensors.Helpers.RandomHelper.CreateSeededRandom(seed));
        _trainingRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(seed + 1);
        if (!_options.RandomBasesPerForward)
            _nmfBases = _network.RandomBases(1, Tensors.Helpers.RandomHelper.CreateSeededRandom(seed + 2));
        Layers.AddRange(layers);
        _encoderLayerEnd = layers.Count;
    }

    /// <summary>Loads an mmsegmentation SegNeXt checkpoint's state dict (<c>backbone.*</c>, <c>decode_head.*</c>).</summary>
    internal void LoadTorchWeights(Func<string, int[], double[]> read) => Network.LoadTorchWeights(read);

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input) => _useNativeMode ? Forward(input) : PredictOnnx(input);

    /// <summary>The class logits at the image's size: the head's 1/8-scale logits upsampled bilinearly (align_corners false).</summary>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);
        int h = input.Shape[2], w = input.Shape[3];
        var logits = SpatialResampling.Resize(Engine, Network.Forward(input, IsTrainingMode, _trainingRandom, Bases(input.Shape[0])), h, w);
        return hasBatch ? logits : RemoveBatchDimension(logits);
    }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => Forward(input);

    // The fixed bases for a batch of b (repeated, as bases.repeat(B, 1, 1)), or null to draw them.
    private Tensor<T>? Bases(int batch) => _nmfBases is null ? null : Engine.TensorTile(_nmfBases, new[] { batch, 1, 1 });

    /// <summary>
    /// One step of the paper's training: the mean pixel cross-entropy of the logits resized to the labels (mmseg's
    /// loss_by_feat) against one-hot (or soft) targets [B, classes, H, W].
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
        int height = target.Shape[2], width = target.Shape[3];
        SetTrainingMode(true);
        try
        {
            TrainWithCustomLoss(input, _ =>
                PixelCrossEntropy(SpatialResampling.Resize(Engine, network.Forward(input, training: true, _trainingRandom, Bases(input.Shape[0])), height, width), target),
                Optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
        return true;
    }

    private Tensor<T> PixelCrossEntropy(Tensor<T> logits, Tensor<T> target)
    {
        var shape = logits.Shape.ToArray();
        // Subtracting a per-pixel constant leaves the softmax unchanged; the maximum is taken without a gradient.
        var maximum = new Tensor<T>(new[] { shape[0], 1, shape[2], shape[3] });
        int pixels = shape[2] * shape[3];
        for (int b = 0; b < shape[0]; b++)
            for (int p = 0; p < pixels; p++)
            {
                double m = double.NegativeInfinity;
                for (int c = 0; c < shape[1]; c++) m = Math.Max(m, NumOps.ToDouble(logits[(b * shape[1] + c) * pixels + p]));
                maximum[b * pixels + p] = NumOps.FromDouble(m);
            }
        var centered = Engine.TensorSubtract(logits, Engine.TensorBroadcastTo(maximum, shape));
        var logSumExp = Engine.TensorLog(Engine.ReduceSum(Engine.TensorExp(centered), new[] { 1 }, false));
        var picked = Engine.ReduceSum(Engine.TensorMultiply(centered, target), new[] { 1 }, false);
        var mass = Engine.ReduceSum(target, new[] { 1 }, false);
        return Engine.ReduceMean(Engine.TensorSubtract(Engine.TensorMultiply(logSumExp, mass), picked), new[] { 0, 1, 2 }, false);
    }

    /// <summary>
    /// The paper's AdamW (betas 0.9, 0.999) at <see cref="SegNeXtOptions.LearningRate"/> with weight decay
    /// <see cref="SegNeXtOptions.WeightDecay"/>; the decode head trains at <see cref="SegNeXtOptions.HeadLearningRateMultiplier"/>
    /// times the rate and parameters whose official names contain "norm" take no decay (mmengine's custom keys match
    /// the longest key first and then alphabetically, so "head" wins over "norm"). The rate warms up linearly from
    /// <see cref="SegNeXtOptions.WarmupStartFactor"/> of itself over <see cref="SegNeXtOptions.WarmupIterations"/>, then
    /// decays as a poly of power 1 to zero at <see cref="SegNeXtOptions.TrainingIterations"/>.
    /// </summary>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer()
    {
        var o = _options;
        var optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(null, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = o.LearningRate,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8,
            WeightDecay = o.WeightDecay,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
            EnableGradientClipping = false,
            LearningRateScheduler = new LearningRateSchedulers.LinearWarmupScheduler(o.LearningRate, o.WarmupIterations,
                o.TrainingIterations, o.LearningRate * o.WarmupStartFactor, LearningRateSchedulers.LinearWarmupScheduler.DecayMode.Linear, 0.0),
            SchedulerStepMode = LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
        });
        var network = Network;
        var head = new HashSet<LayerBase<T>>(network.HeadLayers());
        var normNamed = new HashSet<LayerBase<T>>(network.NormNamedLayers());
        var parameters = new List<Tensor<T>>();
        var rates = new List<T>();
        var decays = new List<T>();
        foreach (var layer in network.OwnedLayers)
        {
            var rate = NumOps.FromDouble(head.Contains(layer) ? o.HeadLearningRateMultiplier : 1.0);
            var decay = NumOps.FromDouble(normNamed.Contains(layer) ? 0.0 : 1.0);
            foreach (var p in Training.TapeTrainingStep<T>.CollectParameters(new ILayer<T>[] { layer }))
            {
                parameters.Add(p);
                for (int i = 0; i < p.Length; i++)
                {
                    rates.Add(rate);
                    decays.Add(decay);
                }
            }
        }
        var options = (AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>)optimizer.GetOptions();
        options.LearningRateMask = new Vector<T>(rates.ToArray());
        options.WeightDecayMask = new Vector<T>(decays.ToArray());
        options.MaskParameters = parameters;
        return optimizer;
    }

    /// <inheritdoc/>
    protected override void ResolveLazyLayerShapes()
    {
    }

    #region ONNX

    protected override Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        bool hasBatch = input.Rank == 4;
        if (!hasBatch)
        {
            input = AddBatchDimension(input);
        }

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            inputData[i] = Convert.ToSingle(input.Data.Span[i]);
        }

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
        var inputMeta = _onnxSession.InputMetadata;

        string inputName = inputMeta.Keys.FirstOrDefault() ?? "pixel_values";
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, onnxInput)
        };

        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputShape = outputTensor.Dimensions.ToArray();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
        {
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

        var result = new Tensor<T>(outputShape, new Vector<T>(outputData));

        if (!hasBatch)
        {
            result = RemoveBatchDimension(result);
        }

        return result;
    }

    #endregion

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var additionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "SegNeXt" },
            { "Description", "SegNeXt semantic segmentation (Guo et al. 2022): MSCAN encoder and LightHamHead decoder" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "InputChannels", _channels },
            { "NumClasses", _numClasses },
            { "ModelSize", _options.ModelSize.ToString() },
            { "UseNativeMode", _useNativeMode },
            { "NumLayers", Layers.Count },
        };
        return new ModelMetadata<T>
        {
            AdditionalInfo = additionalInfo,
            ModelDataProvider = () => SerializeForMetadata()
        };
    }
}
