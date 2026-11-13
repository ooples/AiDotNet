using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.Matting;

/// <summary>
/// RVM: Robust Video Matting for real-time human segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> RVM extracts people from video backgrounds in real-time.
/// Unlike simple background removal that creates hard edges, matting produces
/// a soft alpha matte that preserves hair details and semi-transparent regions.
///
/// Key capabilities:
/// - Real-time video matting without green screen
/// - High-quality alpha matte output
/// - Temporal consistency across frames
/// - Works with any background
///
/// Outputs:
/// - Alpha matte: Transparency at each pixel (0=background, 1=foreground)
/// - Foreground: RGB colors of the person with pre-multiplied alpha
///
/// Example usage:
/// <code>
/// var model = new RVM&lt;double&gt;(arch);
/// var (alpha, foreground) = model.MatteSingleFrame(frame);
/// var composite = model.CompositeWithBackground(foreground, alpha, newBackground);
/// </code>
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - A four-stage stride-2 encoder (a ReLU6 conv stack standing in for MobileNetV3) with an LR-ASPP head
/// - The paper's recurrent decoder: a ConvGRU at the bottleneck and in each of the three upsampling blocks, on half
///   of the channels, carrying temporal state from frame to frame (reset with <see cref="ResetState"/>)
/// - Output: alpha and a foreground residual added to the source frame, both clamped to [0, 1]
/// - The deep guided filter for high-resolution refinement is not included; frames are matted at input resolution
/// </para>
/// <para>
/// <b>Reference:</b> "Robust High-Resolution Video Matting with Temporal Guidance"
/// https://arxiv.org/abs/2108.11515
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Robust High-Resolution Video Matting with Temporal Guidance",
    "https://arxiv.org/abs/2108.11515",
    Year = 2021,
    Authors = "Shanchuan Lin, Linjie Yang, Imran Saleemi, Soumyadip Sengupta")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Frames, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Frames, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public partial class RVM<T> : NeuralNetworkBase<T>
{
    private readonly RVMOptions _options;

    #region Fields

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly string? _onnxModelPath;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly int _numFeatures;
    private readonly int _imageHeight;
    private readonly int _imageWidth;

    // ConvGRU hidden states (bottleneck, 1/8, 1/4, 1/2) carried from frame to frame; null entries start at zero.
    [Scratch]
    [KeepsActivation("The ConvGRU hidden states are the recurrent input of the next frame (Lin et al. 2021); they are copied out of the engine buffers and cleared by ResetState.")]
    private Tensor<T>?[] _recurrentState = new Tensor<T>?[RvmRecurrentNetwork<T>.StateCount];

    #endregion

    #region Properties

    internal bool UseNativeMode => _useNativeMode;
    /// <inheritdoc />
    public override bool SupportsTraining => _useNativeMode;
    internal int NumFeatures => _numFeatures;
    internal int ImageHeight => _imageHeight;
    internal int ImageWidth => _imageWidth;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public RVM()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.ThreeDimensional,
            taskType: Enums.NeuralNetworkTaskType.BinaryClassification,
            inputHeight: 256, inputWidth: 256, inputDepth: 3,
            outputSize: 1))
    {
    }

    public RVM(
        NeuralNetworkArchitecture<T> architecture,
        RVMOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture: architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new RVMOptions();
        _options.Validate();
        Options = _options;

        _useNativeMode = true;
        _numFeatures = _options.NumFeatures;
        _imageHeight = architecture.InputHeight > 0 ? architecture.InputHeight : 512;
        _imageWidth = architecture.InputWidth > 0 ? architecture.InputWidth : 512;

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    public RVM(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        RVMOptions? options = null)
        : base(architecture: architecture, new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new RVMOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"RVM ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        // Validated after the path check so a missing model file reports itself as
        // FileNotFoundException rather than being pre-empted by the options.
        _options.Validate();

        _numFeatures = _options.NumFeatures;
        _imageHeight = architecture.InputHeight > 0 ? architecture.InputHeight : 512;
        _imageWidth = architecture.InputWidth > 0 ? architecture.InputWidth : 512;
        _lossFunction = new MeanSquaredErrorLoss<T>();

        try { _onnxSession = new InferenceSession(onnxModelPath); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load ONNX model: {ex.Message}", ex); }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Processes a video to extract alpha mattes and foregrounds.
    /// </summary>
    public List<(Tensor<T> Alpha, Tensor<T> Foreground)> MatteVideo(List<Tensor<T>> frames)
    {
        var results = new List<(Tensor<T>, Tensor<T>)>();
        ResetState();

        foreach (var frame in frames)
        {
            var (alpha, foreground) = MatteSingleFrame(frame);
            results.Add((alpha, foreground));
        }

        return results;
    }

    /// <summary>
    /// Mattes a single frame, maintaining temporal consistency with previous frames.
    /// </summary>
    public (Tensor<T> Alpha, Tensor<T> Foreground) MatteSingleFrame(Tensor<T> frame)
    {
        var output = _useNativeMode ? Forward(frame) : PredictOnnx(frame);

        // Split output into alpha (1 channel) and foreground RGB (3 channels)
        int c = output.Rank == 4 ? output.Shape[1] : output.Shape[0];
        int h = output.Rank == 4 ? output.Shape[2] : output.Shape[1];
        int w = output.Rank == 4 ? output.Shape[3] : output.Shape[2];

        var alpha = new Tensor<T>([1, 1, h, w]);
        var foreground = new Tensor<T>([1, 3, h, w]);

        // First channel is alpha
        output.Data.Span.Slice(0, h * w).CopyTo(alpha.Data.Span);

        // Remaining 3 channels are foreground RGB
        output.Data.Span.Slice(h * w, 3 * h * w).CopyTo(foreground.Data.Span);

        return (alpha, foreground);
    }

    /// <summary>
    /// Composites the foreground onto a new background.
    /// </summary>
    public Tensor<T> CompositeWithBackground(Tensor<T> foreground, Tensor<T> alpha, Tensor<T> background)
    {
        if (foreground is null) throw new ArgumentNullException(nameof(foreground));
        if (alpha is null) throw new ArgumentNullException(nameof(alpha));
        if (background is null) throw new ArgumentNullException(nameof(background));
        if (foreground.Rank != 3 && foreground.Rank != 4)
            throw new ArgumentException("The foreground must be [C, H, W] or [B, C, H, W].", nameof(foreground));

        // comp = a * fg + (1 - a) * bg = bg + a * (fg - bg), for every sample and every channel. Compositing only the
        // first sample's RGB returned zeros for the rest of a batch.
        bool batched = foreground.Rank == 4;
        var fg = batched ? foreground : Engine.Reshape(foreground, [1, foreground.Shape[0], foreground.Shape[1], foreground.Shape[2]]);
        int batch = fg.Shape[0], channels = fg.Shape[1], h = fg.Shape[2], w = fg.Shape[3];
        int[] full = [batch, channels, h, w];

        var a = Engine.TensorBroadcastTo(Fit(alpha, 1, nameof(alpha)), full);
        var bg = Engine.TensorBroadcastTo(Fit(background, channels, nameof(background)), full);
        var composite = Engine.TensorAdd(bg, Engine.TensorMultiply(a, Engine.TensorSubtract(fg, bg)));
        return batched ? composite : Engine.Reshape(composite, [channels, h, w]);

        // A rank-3 or rank-4 tensor as [batch or 1, channels, h, w]: the leading channels and the top-left h x w,
        // so a larger background is cropped as before.
        Tensor<T> Fit(Tensor<T> t, int wanted, string name)
        {
            if (t.Rank != 3 && t.Rank != 4)
                throw new ArgumentException($"{name} must be [C, H, W] or [B, C, H, W].", name);
            var t4 = t.Rank == 4 ? t : Engine.Reshape(t, [1, t.Shape[0], t.Shape[1], t.Shape[2]]);
            if ((t4.Shape[0] != 1 && t4.Shape[0] != batch) || t4.Shape[1] < wanted || t4.Shape[2] < h || t4.Shape[3] < w)
            {
                throw new ArgumentException(
                    $"{name} shaped [{string.Join(", ", t.Shape.ToArray())}] does not cover {batch} sample(s) of " +
                    $"{wanted} channel(s) at {h}x{w}.", name);
            }

            return Engine.TensorNarrow(Engine.TensorNarrow(Engine.TensorNarrow(t4, 1, 0, wanted), 2, 0, h), 3, 0, w);
        }
    }

    /// <summary>
    /// Resets the recurrent state for a new video.
    /// </summary>
    public new void ResetState()
    {
        _recurrentState = new Tensor<T>?[RvmRecurrentNetwork<T>.StateCount];
        base.ResetState();
    }

    /// <summary>
    /// Extracts just the alpha matte.
    /// </summary>
    public Tensor<T> GetAlpha(Tensor<T> frame)
    {
        var (alpha, _) = MatteSingleFrame(frame);
        return alpha;
    }

    /// <summary>
    /// Creates a green screen effect (extracts foreground).
    /// </summary>
    public Tensor<T> GreenScreenExtract(Tensor<T> frame)
    {
        var (alpha, foreground) = MatteSingleFrame(frame);

        int h = foreground.Rank == 4 ? foreground.Shape[2] : foreground.Shape[1];
        int w = foreground.Rank == 4 ? foreground.Shape[3] : foreground.Shape[2];

        // Create green background
        var greenBg = Engine.TensorConcatenate(new[]
        {
            new Tensor<T>([1, 1, h, w]),
            Tensor<T>.CreateDefault([1, 1, h, w], NumOps.One), // Green
            new Tensor<T>([1, 1, h, w]),
        }, 1);

        return CompositeWithBackground(foreground, alpha, greenBg);
    }

    #endregion

    #region Inference

    /// <summary>
    /// Runs the recurrent network on a frame <c>[C, H, W]</c> or <c>[B, C, H, W]</c>, or on a clip
    /// <c>[B, T, C, H, W]</c> frame by frame, carrying the ConvGRU states from the previous call; returns
    /// alpha then foreground RGB in the input's layout.
    /// </summary>
    private Tensor<T> Forward(Tensor<T> input)
    {
        var (output, state) = Run(input, _recurrentState);
        _recurrentState = state;
        return output;
    }

    /// <inheritdoc />
    /// <remarks>Each training sample starts from a zero recurrent state and leaves the inference state alone; a clip
    /// <c>[B, T, C, H, W]</c> is unrolled through time, as the paper trains on sequences.</remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
        => _useNativeMode ? Run(input, null).Output : base.ForwardForTraining(input);

    private (Tensor<T> Output, Tensor<T>?[] State) Run(Tensor<T> input, Tensor<T>?[]? state)
    {
        var network = new RvmRecurrentNetwork<T>(Engine, Layers);
        var current = state ?? new Tensor<T>?[RvmRecurrentNetwork<T>.StateCount];

        if (input.Rank == 5)
        {
            int batch = input.Shape[0], frames = input.Shape[1];
            var outputs = new Tensor<T>[frames];
            for (int t = 0; t < frames; t++)
            {
                var frame = Engine.Reshape(Engine.TensorNarrow(input, 1, t, 1),
                    [batch, input.Shape[2], input.Shape[3], input.Shape[4]]);
                var (matte, next) = network.Forward(frame, current);
                outputs[t] = matte;
                current = next;
            }

            return (Engine.TensorStack(outputs, 1), Detached(current));
        }

        bool batched = input.Rank == 4;
        var source = batched ? input : Engine.Reshape(input, [1, input.Shape[0], input.Shape[1], input.Shape[2]]);
        var (output, nextState) = network.Forward(source, current);
        return (batched ? output : Engine.Reshape(output, [output.Shape[1], output.Shape[2], output.Shape[3]]),
            Detached(nextState));
    }

    // The hidden states are kept across calls, so they are copied out of the engine's buffers.
    private static Tensor<T>?[] Detached(IReadOnlyList<Tensor<T>?> state)
        => state.Select(h => h?.Clone()).ToArray();

    private Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null) throw new InvalidOperationException("ONNX session is not initialized.");

        var inputData = TensorCopyHelper.CastLogical<T, float>(input).ToArray();

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_onnxSession.InputMetadata.Keys.First(), onnxInput) };

        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputData = new Tensor<float>(outputTensor.ToDenseTensor().Buffer.ToArray(), [(int)outputTensor.Length])
            .Cast<T>().ToArray();

        return new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input) => _useNativeMode ? Forward(input) : PredictOnnx(input);

    #endregion

    #region Serialization

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) { ClearLayers(); return; }

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
            Layers.AddRange(Architecture.Layers);
        else
        {
            int ch = Architecture.InputDepth > 0 ? Architecture.InputDepth : 3;
            Layers.AddRange(LayerHelper<T>.CreateDefaultRVMLayers(ch, _imageHeight, _imageWidth, _numFeatures));
        }

        // Bind once here so a custom layer list that does not follow CreateDefaultRVMLayers fails at construction.
        _ = new RvmRecurrentNetwork<T>(Engine, Layers);
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "RVM" }, { "NumFeatures", _numFeatures },
            { "ImageHeight", _imageHeight }, { "ImageWidth", _imageWidth }
        },
        ModelDataProvider = () => _useNativeMode ? this.Serialize() : []
    };





    #endregion
}
