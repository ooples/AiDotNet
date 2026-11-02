using AiDotNet.LearningRateSchedulers;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// BasicVSR: baseline bidirectional recurrent video super-resolution.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The network is the authors' BasicVSRNet (open-mmlab/mmagic, models/editors/basicvsr/basicvsr_net.py). SPyNet
/// estimates the flow between neighbouring low-resolution frames. A backward branch runs from the last frame to the
/// first: the previous state is warped by the flow to the next frame, concatenated with the current frame and refined
/// by a 3x3 convolution and 30 residual blocks. A forward branch does the same from the first frame to the last, and
/// at each frame the two states are fused by a 1x1 convolution and reconstructed by two pixel-shuffle x2 stages onto
/// the bilinearly upsampled frame, so the output is four times the input's height and width.
/// </para>
/// <para>
/// Training follows configs/basicvsr/basicvsr_2xb4_reds4.py: Charbonnier loss, Adam (0.9, 0.99) at 2e-4 with SPyNet
/// at 0.125 times the rate and frozen for the first 5,000 iterations, cosine-annealed to 1e-7 over 300,000. The release
/// loads a pretrained SPyNet; this native model starts it from PyTorch's default initialization.
/// </para>
/// <para>
/// <b>For Beginners:</b> BasicVSR processes video frames like reading a book forward and
/// backward simultaneously. By looking in both directions, each frame gets context from
/// past AND future frames. Optical flow tells the model how pixels moved between frames,
/// so it can properly align them before combining their information.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputType: InputType.FourDimensional,
///     taskType: NeuralNetworkTaskType.Regression, inputFrames: 15, inputHeight: 64, inputWidth: 64, inputDepth: 3);
/// var model = new BasicVSR&lt;float&gt;(arch, new BasicVSROptions());
/// model.Train(lowResClip, highResClip);         // [15, 3, 64, 64] -> [15, 3, 256, 256], pixel values 0-255
/// var upscaled = model.Upscale(lowResClip);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond",
    "https://arxiv.org/abs/2012.02181",
    Year = 2021,
    Authors = "Kelvin C.K. Chan, Xintao Wang, Ke Yu, Chao Dong, Chen Change Loy")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 2e-4, Beta2 = 0.99,
                Schedule = LearningRateSchedulerType.CosineAnnealing, MinLearningRate = 1e-7,
                Source = "Chan et al. 2021, Sec. 4: Adam with cosine annealing; configs/basicvsr/basicvsr_2xb4_reds4.py: "
                        + "lr 2e-4, betas (0.9, 0.99), SPyNet lr_mult 0.125 and frozen for fix_iter = 5000, "
                        + "CosineRestartLR over 300k iterations to eta_min 1e-7.")]
public partial class BasicVSR<T> : VideoSuperResolutionBase<T>
{
    #region Fields

    private readonly BasicVSROptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _usingCustomLayers;
    private bool _disposed;
    private int _trainingSteps;

    private BasicVsrBackbone<T>? _backbone;

    #endregion

    #region Constructors

    /// <summary>Creates a BasicVSR model in ONNX inference mode.</summary>
    public BasicVSR(NeuralNetworkArchitecture<T> architecture, string modelPath, BasicVSROptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options ?? new BasicVSROptions();
        Options = _options;
        _useNativeMode = false;
        ScaleFactor = 4;
        NumFrames = _options.NumFrames;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a BasicVSR model in native training mode.</summary>
    public BasicVSR(NeuralNetworkArchitecture<T> architecture, BasicVSROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new CharbonnierLoss<T>((options ?? new BasicVSROptions()).CharbonnierEpsilon))
    {
        _options = options ?? new BasicVSROptions();
        Options = _options;
        if (_options.MidChannels < 1) throw new ArgumentOutOfRangeException(nameof(options), "MidChannels must be positive.");
        if (_options.NumResBlocks < 0) throw new ArgumentOutOfRangeException(nameof(options), "NumResBlocks cannot be negative.");
        _useNativeMode = true;
        ScaleFactor = 4;
        NumFrames = _options.NumFrames;
        InitializeLayers();
        _optimizer = optimizer;
    }

    #endregion

    #region Video Super-Resolution

    /// <inheritdoc />
    public override Tensor<T> Upscale(Tensor<T> lowResFrames)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessFrames(lowResFrames);
        var output = IsOnnxMode ? RunOnnxInference(preprocessed) : Forward(preprocessed);
        return PostprocessOutput(output);
    }

    #endregion

    #region Layers

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            _usingCustomLayers = true;
            Layers.AddRange(Architecture.Layers);
            return;
        }

        _backbone = new BasicVsrBackbone<T>(Layers, _options.MidChannels, _options.NumResBlocks);
    }

    /// <inheritdoc />
    protected override void ResolveLazyLayerShapes()
    {
        if (_usingCustomLayers || !_useNativeMode) return;
        using var noGrad = new NoGradScope<T>();
        Network(new Tensor<T>(new[] { 1, 2, 3, 32, 32 }));
    }

    #endregion

    #region Network

    private BasicVsrBackbone<T> Backbone
        => _backbone ?? throw new InvalidOperationException("BasicVSR's native layers are not initialized.");

    /// <summary>BasicVSRNet.forward: clips [N, T, 3, H, W] in [0, 1] to [N, T, 3, 4H, 4W].</summary>
    private Tensor<T> Network(Tensor<T> clips) => Backbone.Run(Engine, Layers, clips);

    private Tensor<T> RunClips(Tensor<T> frames)
    {
        bool batched = frames.Rank == 5;
        var clips = batched ? frames : Engine.Reshape(frames, new[] { 1 }.Concat(frames._shape).ToArray());
        var output = Network(clips);
        return batched ? output : Engine.Reshape(output, output._shape.Skip(1).ToArray());
    }

    /// <summary>Test hook: SPyNet's flow from <paramref name="reference"/> to <paramref name="support"/>, [N, 3, H, W] in [0, 1].</summary>
    internal Tensor<T> FlowForTest(Tensor<T> reference, Tensor<T> support)
        => Backbone.SpyNet.Flow(Engine, Layers, reference, support);

    /// <summary>Test hook: mmagic's flow_warp with zero padding.</summary>
    internal Tensor<T> WarpForTest(Tensor<T> image, Tensor<T> flow) => BasicVsrBlocks.FlowWarp(Engine, image, flow, GridSamplePadding.Zeros);

    /// <summary>Test hook: the network on frames already scaled to [0, 1].</summary>
    internal Tensor<T> UpscaleNormalizedForTest(Tensor<T> frames) => RunClips(frames);

    /// <summary>The SPyNet parameters.</summary>
    internal IReadOnlyList<Tensor<T>> FlowParameters
        => _backbone is null ? Array.Empty<Tensor<T>>() : _backbone.FlowParameters(Layers);

    #endregion

    #region Forward and training

    /// <inheritdoc />
    protected override Tensor<T> Forward(Tensor<T> input)
        => _usingCustomLayers ? base.Forward(input) : RunClips(input);

    /// <inheritdoc />
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => _usingCustomLayers ? base.ForwardForTraining(input) : RunClips(PreprocessFrames(input));

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> BuildPaperOptimizer()
    {
        if (_optimizer is not null) return _optimizer;
        // The rate mask is keyed by parameter tensor, so the lazily shaped convolutions must exist first.
        ResolveLazyLayerShapes();
        // Adam is AdamW without decay; AdamW carries the per-parameter rate mask that gives SPyNet its 0.125.
        var optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            Beta1 = 0.9,
            Beta2 = 0.99,
            Epsilon = 1e-8,
            WeightDecay = 0.0,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
            EnableGradientClipping = false,
            LearningRateScheduler = new CosineAnnealingLRScheduler(
                _options.LearningRate, Math.Max(1, _options.CosineIterations), _options.MinLearningRate),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        });
        var flow = new HashSet<Tensor<T>>(FlowParameters);
        var parameters = TapeTrainingStep<T>.CollectParameters(Layers);
        var rates = new List<T>();
        foreach (var p in parameters)
        {
            var rate = NumOps.FromDouble(flow.Contains(p) ? _options.SpyNetLearningRateMultiplier : 1.0);
            for (int i = 0; i < p.Length; i++) rates.Add(rate);
        }
        var options = (AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>)optimizer.GetOptions();
        options.LearningRateMask = new Vector<T>(rates.ToArray());
        options.MaskParameters = parameters;
        _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, optimizer);
        return _optimizer;
    }

    private IReadOnlyList<Tensor<T>> TrainableParameters()
    {
        var all = TapeTrainingStep<T>.CollectParameters(Layers);
        if (_trainingSteps > _options.FlowFreezeIterations) return all;
        var flow = new HashSet<Tensor<T>>(FlowParameters);
        return all.Where(p => !flow.Contains(p)).ToList();
    }

    /// <inheritdoc />
    /// <remarks>Charbonnier loss over every output frame; SPyNet joins training after the freeze iterations.</remarks>
    protected override IReadOnlyList<TrainingObjective<T>>? DefineTrainingObjectives(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (_usingCustomLayers || !_useNativeMode) return null;
        _trainingSteps++;
        var optimizer = BuildPaperOptimizer();
        return new[]
        {
            new TrainingObjective<T>(new[] { input, expectedOutput }, data =>
                LossFunction.ComputeTapeLoss(RunClips(PreprocessFrames(data[0])), NormalizeFrames(data[1])))
            {
                Optimizer = optimizer,
                Parameters = TrainableParameters,
            }
        };
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _useNativeMode && !_usingCustomLayers ? BuildPaperOptimizer() : base.GetOrCreateBaseOptimizer();

    #endregion

    #region NeuralNetworkBase

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames) => NormalizeFrames(rawFrames);

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => DenormalizeFrames(modelOutput);

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "BasicVSR-Native" : "BasicVSR-ONNX",
            Description = "BasicVSR bidirectional recurrent VSR (Chan et al., CVPR 2021)",
            Complexity = _options.NumResBlocks
        };
        m.AdditionalInfo["MidChannels"] = _options.MidChannels.ToString();
        m.AdditionalInfo["NumResBlocks"] = _options.NumResBlocks.ToString();
        m.AdditionalInfo["ScaleFactor"] = "4";
        return m;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(BasicVSR<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing)
        {
            OnnxModel?.Dispose();
        }
        base.Dispose(disposing);
    }

    #endregion
}
