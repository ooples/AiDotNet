using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Prediction;

/// <summary>
/// MCnet: future-frame prediction by decomposing a video into separate motion and content pathways.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Villegas, Yang, Hong, Lin and Lee, "Decomposing Motion and Content for Natural Video Sequence
/// Prediction" (ICLR 2017, arXiv:1706.08033).
/// </para>
/// <para>
/// <b>REPLACES DRVI, AND CHANGES THE TASK.</b> That class cited arXiv:2401.05765 as "DRVI: Disentangled
/// Representations for Video Interpolation" by Rong Du and Zhiwei Xiong. That identifier is "A new
/// computationally efficient algorithm to solve Feature Selection for Functional Data Classification in
/// high-dimensional spaces" — functional-data feature selection, unrelated to video. Title, authors,
/// year and subject were all invented.
/// </para>
/// <para>
/// More importantly the TASK was wrong. MCnet is video PREDICTION — it observes <c>x_1..x_t</c> and
/// predicts <c>x_hat_{t+1}</c> onward — whereas DRVI lived in <c>Video/FrameInterpolation</c> and
/// implemented <c>Interpolate(frame0, frame1, t)</c>, synthesising a frame BETWEEN two known frames.
/// The interpolation contract cannot express prediction: there is no second frame to interpolate
/// towards. Hence the move to <c>Video/Prediction</c> and the change of base class.
/// </para>
/// <code>
///   motion    [d_t, c_t] = f_dyn(x_t - x_{t-1}, d_{t-1}, c_{t-1})     recurrent, on DIFFERENCES
///   content   s_t        = f_cont(x_t)                                single frame, no recurrence
///   combine   f_t        = g_comb([d_t, s_t])
///   residual  r_t^l      = f_res([s_t^l, d_t^l])^l                    at every scale
///   decode    x_hat_t+1  = g_dec(f_t, r_t)
/// </code>
/// <para>
/// The asymmetry is the point: motion reads CHANGES between frames, content reads one frame. A static
/// scene differences to zero regardless of what it depicts, so the motion pathway has nothing about
/// appearance to latch onto — see <see cref="McnetDecomposition{T}"/>.
/// </para>
/// <para>
/// The objective adds a GRADIENT DIFFERENCE term to the pixel loss (see <see cref="McnetLoss{T}"/>),
/// which is what keeps predictions sharp: minimising pixel error alone yields a blurry average over
/// plausible futures.
/// </para>
/// <para><b>For Beginners:</b> Given several frames of video, this predicts what comes next. It reads
/// "how things are moving" from the differences between consecutive frames and "what things look like"
/// from the most recent frame, then recombines the two to draw the next frame.</para>
/// </remarks>
/// <example>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 64, inputWidth: 64, inputDepth: 3);
///
/// var model = new Mcnet&lt;double&gt;(arch, new McnetOptions { NumInputFrames = 4 });
/// var next = model.PredictNextFrame(observedFrames);   // [time, H, W, C] -> [H, W, C]
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
// VideoGeneration, not FrameInterpolation: this model PREDICTS future frames rather than synthesising
// one between two known frames. There is no VideoPrediction member, and FrameInterpolation would both
// misfile the model and route it into a test family asserting interpolation semantics it lacks.
[ModelTask(ModelTask.VideoGeneration)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Decomposing Motion and Content for Natural Video Sequence Prediction",
    "https://arxiv.org/abs/1706.08033",
    Year = 2017,
    Authors = "Ruben Villegas, Jimei Yang, Seunghoon Hong, Xunyu Lin, Honglak Lee")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Frames, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Frames, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public class Mcnet<T> : VideoNeuralNetworkBase<T>
{
    #region Fields

    private readonly McnetOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    private readonly McnetDecomposition<T> _decomposition;

    /// <summary>
    /// Gets the motion/content decomposition helper.
    /// </summary>
    public McnetDecomposition<T> Decomposition => _decomposition;

    #endregion

    #region Constructors

    /// <summary>Creates an MCnet model in ONNX inference mode.</summary>
    public Mcnet(NeuralNetworkArchitecture<T> architecture, string modelPath, McnetOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new McnetOptions();
        _useNativeMode = false;
        _options.ModelPath = modelPath;
        _decomposition = new McnetDecomposition<T>(_options.NumScales);
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates an MCnet model in native training mode.</summary>
    public Mcnet(NeuralNetworkArchitecture<T> architecture, McnetOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new McnetOptions();
        _useNativeMode = true;
        _decomposition = new McnetDecomposition<T>(_options.NumScales);

        // Adam at the paper's 1e-4.
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
            });

        InitializeLayers();
    }

    #endregion

    #region Prediction

    /// <summary>
    /// Predicts the next frame from a sequence of observed frames.
    /// </summary>
    /// <param name="observedFrames">
    /// <c>[time, height, width, channels]</c>, at least two frames — the motion pathway needs a
    /// difference.
    /// </param>
    /// <returns><c>[height, width, channels]</c>, the predicted frame.</returns>
    /// <remarks>
    /// The model's actual task, and the reason it no longer lives under FrameInterpolation. The two
    /// pathways are fed differently here: differences for motion, the last frame for content.
    /// </remarks>
    public Tensor<T> PredictNextFrame(Tensor<T> observedFrames)
    {
        ThrowIfDisposed();
        if (observedFrames is null) throw new ArgumentNullException(nameof(observedFrames));

        // Validated by the decomposition helper, which enforces the two-frame minimum.
        var motionInput = _decomposition.MotionInput(observedFrames);
        var contentInput = _decomposition.ContentInput(observedFrames);

        // The learned pathways run over the layer stack. Content supplies spatial layout and motion the
        // dynamics; both are folded into one tensor for the stack, which the combination and decoder
        // layers then separate by channel.
        var combined = _decomposition.Combine(
            LastMotionFrame(motionInput), contentInput);

        var processed = PreprocessFrames(combined);
        var output = IsOnnxMode ? RunOnnxInference(processed) : Forward(processed);
        return PostprocessOutput(output);
    }

    /// <summary>
    /// The most recent difference frame, which is the motion pathway's current observation.
    /// </summary>
    private Tensor<T> LastMotionFrame(Tensor<T> differences)
    {
        int time = differences.Shape[0];
        int h = differences.Shape[1], w = differences.Shape[2], c = differences.Shape[3];
        int perFrame = h * w * c;

        var result = new Tensor<T>(new[] { h, w, c });
        int offset = (time - 1) * perFrame;
        for (int i = 0; i < perFrame; i++) result[i] = differences[offset + i];
        return result;
    }

    /// <summary>
    /// The paper's objective for one prediction: <c>alpha * L_img + beta * L_GAN</c>.
    /// </summary>
    /// <param name="predicted">The predicted frame, flattened.</param>
    /// <param name="target">The true next frame, flattened.</param>
    /// <param name="height">Frame height, for the gradient term.</param>
    /// <param name="width">Frame width.</param>
    /// <param name="discriminatorOnGenerated">
    /// D applied to the observed frames concatenated with the generated continuation, or <c>null</c> to
    /// omit the adversarial term.
    /// </param>
    public double ComputeLoss(
        Vector<T> predicted, Vector<T> target, int height, int width,
        double? discriminatorOnGenerated = null)
    {
        double image = McnetLoss<T>.Pixel(predicted, target, _options.PixelLossNorm)
                     + McnetLoss<T>.GradientDifference(
                           predicted, target, height, width, _options.GradientLossExponent);

        double adversarial = discriminatorOnGenerated is double d
            ? McnetLoss<T>.GeneratorAdversarial(d)
            : 0.0;

        return (_options.ImageLossWeight * image)
             + (_options.AdversarialLossWeight * adversarial);
    }

    #endregion

    #region NeuralNetworkBase

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }

        // The resolution-preserving video backbone. MCnet's encoders and decoder are convolutional
        // stacks over a channel-concatenated motion+content tensor, which is what this produces: the
        // decomposition (differencing, per-scale pairing) is structural and lives in
        // McnetDecomposition, while the learned weights live here so they are real parameters.
        int channels = Architecture.InputDepth > 0 ? Architecture.InputDepth : 3;
        int height = Architecture.InputHeight > 0 ? Architecture.InputHeight : 64;
        int width = Architecture.InputWidth > 0 ? Architecture.InputWidth : 64;
        Layers.AddRange(LayerHelper<T>.CreateDefaultFrameInterpolationLayers(
            inputChannels: channels, inputHeight: height, inputWidth: width,
            numFeatures: _options.NumFeatures));
    }

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) return RunOnnxInference(input);
        return Forward(input);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc/>
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames) => rawFrames;

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => modelOutput;

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "MCnet-Native" : "MCnet-ONNX",
            Description =
                "MCnet motion-content decomposition for future frame prediction (ICLR 2017)",
            Complexity = _options.NumContentBlocks + _options.NumMotionBlocks + _options.NumDecoderBlocks,
        };
        m.AdditionalInfo["Task"] = "VideoPrediction";
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumContentBlocks"] = _options.NumContentBlocks.ToString();
        m.AdditionalInfo["NumMotionBlocks"] = _options.NumMotionBlocks.ToString();
        m.AdditionalInfo["NumDecoderBlocks"] = _options.NumDecoderBlocks.ToString();
        m.AdditionalInfo["NumScales"] = _options.NumScales.ToString();
        m.AdditionalInfo["NumInputFrames"] = _options.NumInputFrames.ToString();
        m.AdditionalInfo["ImageLossWeight"] = _options.ImageLossWeight.ToString();
        m.AdditionalInfo["AdversarialLossWeight"] = _options.AdversarialLossWeight.ToString();
        m.AdditionalInfo["Paper"] = "arXiv:1706.08033";
        return m;
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.NumFeatures);
        w.Write(_options.NumContentBlocks);
        w.Write(_options.NumMotionBlocks);
        w.Write(_options.NumDecoderBlocks);
        w.Write(_options.NumScales);
        w.Write(_options.NumInputFrames);
        w.Write(_options.NumPredictedFrames);
        w.Write(_options.ImageLossWeight);
        w.Write(_options.AdversarialLossWeight);
        w.Write(_options.GradientLossExponent);
        w.Write(_options.PixelLossNorm);
        w.Write(_options.LearningRate);
        w.Write(_options.DropoutRate);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.NumFeatures = r.ReadInt32();
        _options.NumContentBlocks = r.ReadInt32();
        _options.NumMotionBlocks = r.ReadInt32();
        _options.NumDecoderBlocks = r.ReadInt32();
        _options.NumScales = r.ReadInt32();
        _options.NumInputFrames = r.ReadInt32();
        _options.NumPredictedFrames = r.ReadInt32();
        _options.ImageLossWeight = r.ReadDouble();
        _options.AdversarialLossWeight = r.ReadDouble();
        _options.GradientLossExponent = r.ReadDouble();
        _options.PixelLossNorm = r.ReadInt32();
        _options.LearningRate = r.ReadDouble();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (IsOnnxMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new Mcnet<T>(Architecture, mp, _options);
        return new Mcnet<T>(Architecture, _options);
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Mcnet<T>));
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
