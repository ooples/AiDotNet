using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Enums;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>
/// Tacotron 2's spectrogram prediction network (Shen et al. 2018, §2.2): characters to a mel spectrogram through a
/// convolutional and bidirectional-LSTM encoder, location-sensitive attention, and an autoregressive decoder of two
/// zoneout LSTMs with a stop token and a residual post-net.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Encoder: a 512-wide character embedding, 3 convolutions of 512 filters (kernel 5) with batch normalization, ReLU and
/// dropout 0.5, and a bidirectional LSTM of 256 units per direction. Decoder, per frame: a pre-net of 2 bias-free
/// 256-unit ReLU layers whose dropout 0.5 stays on at inference, an attention LSTM of 1024 units, location-sensitive
/// attention (128 wide, 32 location filters of length 31) over the previous and cumulative weights, a decoder LSTM of
/// 1024 units, and linear projections of [decoder output; context] to the frame and the stop token. Both LSTMs use
/// zoneout 0.1 (its expectation at inference). Post-net: 5 convolutions of 512 filters (kernel 5) with tanh but the
/// last, added to the decoder's frames.
/// </para>
/// <para>Training (§3.1): teacher forcing, the summed MSE of the frames before and after the post-net plus the stop
/// token's binary cross-entropy; Adam (0.9, 0.999, ε = 1e-6) at 1e-3, L2 weight 1e-6.</para>
/// <para><b>For Beginners:</b> Tacotron 2 reads text one character at a time and writes a spectrogram one frame at a
/// time, choosing at each step which characters to "look at"; a vocoder turns the spectrogram into audio.</para>
/// </remarks>
/// <example>
/// <code>
/// var tacotron = new Tacotron2&lt;float&gt;(architecture, new Tacotron2Options());
/// tacotron.Train(new TtsTrainingSample&lt;float&gt; { Tokens = characters, Audio = recording });
/// var mel = tacotron.TextToMel("Hello world.");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions",
    "https://arxiv.org/abs/1712.05884",
    Year = 2018,
    Authors = "Shen et al.")]
public partial class Tacotron2<T> : TtsModelBase<T>, IAcousticModel<T>, ITrainingObjectiveProvider<T>
{
    private readonly Tacotron2Options _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    private EmbeddingLayer<T>? _embedding;
    private ConvBatchNormStackLayer<T>? _encoderConvolutions;
    private BidirectionalRecurrentLayer<T>? _encoderLstm;
    private BiasFreeLinearLayer<T>? _prenet1;
    private BiasFreeLinearLayer<T>? _prenet2;
    private LSTMCellLayer<T>? _attentionRnn;
    private LocationSensitiveAttentionLayer<T>? _attention;
    private LSTMCellLayer<T>? _decoderRnn;
    private DenseLayer<T>? _melProjection;
    private DenseLayer<T>? _stopProjection;
    private ConvBatchNormStackLayer<T>? _postnet;

    private Random _regularizationRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(0);

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Creates a Tacotron 2 that runs an exported ONNX graph.</summary>
    public Tacotron2(NeuralNetworkArchitecture<T> architecture, string modelPath, Tacotron2Options? options = null)
        : base(architecture)
    {
        _options = options ?? new Tacotron2Options();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a trainable Tacotron 2.</summary>
    public Tacotron2(
        NeuralNetworkArchitecture<T> architecture,
        Tacotron2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new Tacotron2Options();
        _useNativeMode = true;
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        // §3.1: Adam (0.9, 0.999, ε = 1e-6) at 1e-3 "exponentially decaying to 1e-5 starting after 50,000 iterations",
        // L2 weight 1e-6. The paper gives no rate; halving every 50k steps to the 1e-5 floor is Rayhane-mamah/Tacotron-2's
        // reading. Gradient norms are clipped at 1 (NVIDIA tacotron2 grad_clip_thresh).
        double rate = _options.LearningRate;
        _optimizer = optimizer ?? CreateFixedAdamOptimizer(
            initialLearningRate: rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-6,
            l2RegularizationStrength: _options.WeightDecay,
            learningRateScheduler: new AiDotNet.LearningRateSchedulers.LambdaLRScheduler(rate,
                step => step < 50000 ? 1.0 : Math.Max(1e-2, Math.Pow(0.5, (step - 50000) / 50000.0))),
            maxGradientNorm: 1.0);
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    public new int MelChannels => _options.MelChannels;

    /// <inheritdoc />
    public new int HopSize => _options.HopSize;

    /// <inheritdoc />
    public int FftSize => _options.FftSize;

    /// <inheritdoc />
    public Tensor<T> TextToMel(string text) => Synthesize(text);

    private int DecoderStepLimit => Math.Max(1, (_options.MaxMelLength + _options.OutputsPerStep - 1) / _options.OutputsPerStep);

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }

        var o = _options;
        if (o.EncoderDim % 2 != 0)
            throw new ArgumentException("The encoder width is split over two LSTM directions; it must be even.");
        int mel = o.MelChannels, group = mel * o.OutputsPerStep;
        IActivationFunction<T> identity = new IdentityActivation<T>();
        _embedding = new EmbeddingLayer<T>(o.VocabSize, o.EmbeddingDim);
        _encoderConvolutions = new ConvBatchNormStackLayer<T>(o.EmbeddingDim, Enumerable.Repeat(o.EmbeddingDim, o.NumEncoderLayers).ToArray(),
            5, useTanh: false, linearLast: false, dropoutRate: o.ConvolutionDropout);
        _encoderLstm = new BidirectionalRecurrentLayer<T>(o.EmbeddingDim, o.EncoderDim / 2, RecurrentCellType.Lstm);
        _prenet1 = new BiasFreeLinearLayer<T>(group, o.PrenetDim);
        _prenet2 = new BiasFreeLinearLayer<T>(o.PrenetDim, o.PrenetDim);
        _attentionRnn = new LSTMCellLayer<T>(o.PrenetDim + o.EncoderDim, o.AttentionRnnDim);
        _attention = new LocationSensitiveAttentionLayer<T>(o.AttentionRnnDim, o.EncoderDim, o.AttentionDimension,
            o.AttentionLocationChannels, o.AttentionKernelSize);
        _decoderRnn = new LSTMCellLayer<T>(o.AttentionRnnDim + o.EncoderDim, o.DecoderRnnDim);
        _melProjection = new DenseLayer<T>(group, identity);
        _stopProjection = new DenseLayer<T>(1, identity);
        _melProjection.ResolveShapesOnly(new[] { o.DecoderRnnDim + o.EncoderDim });
        _stopProjection.ResolveShapesOnly(new[] { o.DecoderRnnDim + o.EncoderDim });
        var postnet = Enumerable.Repeat(o.PostnetDim, Math.Max(0, o.PostnetLayers - 1)).Append(mel).ToArray();
        _postnet = new ConvBatchNormStackLayer<T>(mel, postnet, 5, useTanh: true, linearLast: true, dropoutRate: o.ConvolutionDropout);

        AddEncoderDecoderLayers(new List<ILayer<T>> { _embedding, _encoderConvolutions, _encoderLstm }, Array.Empty<ILayer<T>>());
        ComponentLayers.AddRange(new LayerBase<T>[]
        {
            _prenet1, _prenet2, _attentionRnn, _attention, _decoderRnn, _melProjection, _stopProjection, _postnet,
        });
    }

    private bool HasPaperLayers => _attention is not null;

    private Tensor<T> Tokens(Tensor<T> characters)
        => characters.Rank == 2 && characters.Shape[0] == 1 ? Engine.Reshape(characters, new[] { characters.Shape[1] }) : characters;

    // [1, frames, mel] -> [frames, mel].
    private Tensor<T> MelRows(Tensor<T> mel)
        => mel.Rank == 3 ? Engine.Reshape(mel, new[] { mel.Shape[1], mel.Shape[2] }) : mel;

    /// <summary>The encoder (§2.2): <c>[characters, encoder width]</c>.</summary>
    private Tensor<T> Encode(Tensor<T> characters)
    {
        var x = Require(_embedding).Forward(Tokens(characters));
        x = Require(_encoderConvolutions).Forward(x);
        return Require(_encoderLstm).Forward(x);
    }

    private static TLayer Require<TLayer>(TLayer? layer) where TLayer : class
        => layer ?? throw new NotSupportedException("This needs the paper's layers; this model was built from caller-supplied layers.");

    /// <summary>The pre-net (§2.2): 2 bias-free fully connected ReLU layers with dropout 0.5 that stays on at inference,
    /// "to introduce output variation".</summary>
    private Tensor<T> Prenet(Tensor<T> previousFrames)
    {
        var x = AlwaysOnDropout(Engine.ReLU(Require(_prenet1).Forward(previousFrames)));
        return AlwaysOnDropout(Engine.ReLU(Require(_prenet2).Forward(x)));
    }

    private Tensor<T> AlwaysOnDropout(Tensor<T> x)
    {
        double p = _options.PrenetDropout;
        if (p <= 0) return x;
        var mask = new Tensor<T>(x._shape);
        var keep = NumOps.FromDouble(1.0 / (1.0 - p));
        for (int i = 0; i < mask.Length; i++) mask[i] = _regularizationRandom.NextDouble() < p ? NumOps.Zero : keep;
        return Engine.TensorMultiply(x, mask);
    }

    // Zoneout (Krueger et al. 2017), probability 0.1 (§2.2): in training each unit of the LSTM state keeps its previous
    // value with probability p; at inference the expectation p · previous + (1 − p) · new.
    private Tensor<T> Zoneout(Tensor<T> previous, Tensor<T> next)
    {
        double p = _options.ZoneoutProbability;
        if (p <= 0) return next;
        if (!IsTrainingMode)
            return Engine.TensorAdd(Engine.TensorMultiplyScalar(previous, NumOps.FromDouble(p)), Engine.TensorMultiplyScalar(next, NumOps.FromDouble(1 - p)));
        var mask = new Tensor<T>(next._shape);
        for (int i = 0; i < mask.Length; i++) mask[i] = _regularizationRandom.NextDouble() < p ? NumOps.One : NumOps.Zero;
        var change = Engine.TensorAddScalar(Engine.TensorNegate(mask), NumOps.One);
        return Engine.TensorAdd(Engine.TensorMultiply(mask, previous), Engine.TensorMultiply(change, next));
    }

    /// <summary>
    /// The autoregressive decoder (§2.2): pre-net on the previous frame group, attention LSTM, location-sensitive
    /// attention over [previous; cumulative] weights, decoder LSTM, then the frame and stop projections of
    /// [decoder output; context]. Teacher-forced when <paramref name="target"/> <c>[frames, mel]</c> is given (the
    /// previous ground-truth group, all zeros first); otherwise it feeds back its own output and stops at the first stop
    /// probability above the threshold. Returns the frames before the post-net <c>[frames, mel]</c> and the stop logits
    /// <c>[steps]</c>.
    /// </summary>
    private (Tensor<T> Frames, Tensor<T> StopLogits) Decode(Tensor<T> memory, Tensor<T>? target)
    {
        var o = _options;
        var attention = Require(_attention);
        var attentionRnn = Require(_attentionRnn);
        var decoderRnn = Require(_decoderRnn);
        int tokens = memory.Shape[0], r = o.OutputsPerStep, mel = o.MelChannels, groupWidth = mel * r;
        int steps = target is null ? DecoderStepLimit : (target.Shape[0] + r - 1) / r;
        var projectedMemory = attention.ProjectMemory(memory);
        var attentionState = new Tensor<T>(new[] { 1, 2 * o.AttentionRnnDim });
        var decoderState = new Tensor<T>(new[] { 1, 2 * o.DecoderRnnDim });
        var context = new Tensor<T>(new[] { 1, o.EncoderDim });
        var weights = new Tensor<T>(new[] { 1, tokens });
        var cumulative = new Tensor<T>(new[] { 1, tokens });
        var previous = new Tensor<T>(new[] { 1, groupWidth });
        var groups = new List<Tensor<T>>();
        var stops = new List<Tensor<T>>();
        for (int step = 0; step < steps; step++)
        {
            var prenet = Prenet(previous);
            attentionState = Zoneout(attentionState, attentionRnn.Forward(Engine.TensorConcatenate(new[] { prenet, context }, 1), attentionState));
            var attentionHidden = attentionRnn.SplitState(attentionState).Hidden;
            (context, weights) = attention.Attend(attentionHidden, memory, projectedMemory, weights, cumulative);
            cumulative = Engine.TensorAdd(cumulative, weights);
            decoderState = Zoneout(decoderState, decoderRnn.Forward(Engine.TensorConcatenate(new[] { attentionHidden, context }, 1), decoderState));
            var decoderOutput = Engine.TensorConcatenate(new[] { decoderRnn.SplitState(decoderState).Hidden, context }, 1);
            var group = Require(_melProjection).Forward(decoderOutput);                         // [1, mel · r]
            var stop = Require(_stopProjection).Forward(decoderOutput);                         // [1, 1]
            groups.Add(group);
            stops.Add(Engine.Reshape(stop, new[] { 1 }));
            if (target is null)
            {
                previous = group;
                if (1.0 / (1.0 + Math.Exp(-NumOps.ToDouble(stop[0, 0]))) > o.StopThreshold)
                    break;
            }
            else
            {
                previous = TargetGroup(target, step);
            }
        }
        var frames = Engine.Reshape(groups.Count == 1 ? groups[0] : Engine.TensorConcatenate(groups.ToArray(), 0),
            new[] { groups.Count * r, mel });
        var stopLogits = stops.Count == 1 ? stops[0] : Engine.TensorConcatenate(stops.ToArray(), 0);
        return (frames, stopLogits);
    }

    // The ground-truth frame group of step `step`, zero-padded past the end.
    private Tensor<T> TargetGroup(Tensor<T> target, int step)
    {
        int r = _options.OutputsPerStep, mel = _options.MelChannels;
        var group = new Tensor<T>(new[] { 1, mel * r });
        for (int f = 0; f < r; f++)
        {
            int frame = step * r + f;
            if (frame >= target.Shape[0]) break;
            for (int m = 0; m < mel; m++) group[0, f * mel + m] = target[frame, m];
        }
        return group;
    }

    /// <inheritdoc />
    /// <remarks>Synthesis (§2.2): encode, decode autoregressively until the stop token fires, add the post-net residual;
    /// <c>[1, frames, mel]</c>. The pre-net dropout is drawn from the sampling seed, so a model synthesizes the same
    /// output for the same text.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        if (!HasPaperLayers)
        {
            var x = input;
            foreach (var layer in Layers) x = layer.Forward(x);
            return x;
        }
        using var _ = new NoGradScope<T>();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        _regularizationRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        try
        {
            var (frames, _) = Decode(Encode(input), null);
            var refined = Engine.TensorAdd(frames, Require(_postnet).Forward(frames));
            return Engine.Reshape(refined, new[] { 1, refined.Shape[0], _options.MelChannels });
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>
    /// The training objective (§2.2, §3.1; reference <c>Tacotron2Loss</c>): teacher-forced decoding, the summed MSE of
    /// the mel before and after the post-net, plus the binary cross-entropy of the stop logits against a target that is
    /// 1 from the last frame's step on.
    /// </summary>
    private Tensor<T> Objective(Tensor<T> characters, Tensor<T> expected)
    {
        var target = MelRows(expected);
        int frames = target.Shape[0], mel = _options.MelChannels;
        var (decoded, stopLogits) = Decode(Encode(characters), target);
        var before = Engine.TensorSlice(decoded, new[] { 0, 0 }, new[] { frames, mel });
        var after = Engine.TensorAdd(before, Require(_postnet).Forward(before));
        var loss = Engine.TensorAdd(MeanSquaredDifference(before, target), MeanSquaredDifference(after, target));

        int steps = stopLogits.Length;
        var stopTarget = new Tensor<T>(new[] { steps });
        stopTarget[steps - 1] = NumOps.One;
        // BCE with logits: softplus(z) − z · y, averaged over steps.
        var bce = Engine.TensorSubtract(Engine.Softplus(stopLogits), Engine.TensorMultiply(stopLogits, stopTarget));
        return Engine.TensorAdd(loss, Engine.ReduceMean(bce, new[] { 0 }, keepDims: false));
    }

    private Tensor<T> MeanSquaredDifference(Tensor<T> predicted, Tensor<T> target)
    {
        var diff = Engine.TensorSubtract(predicted, target);
        var squared = Engine.TensorMultiply(diff, diff);
        return Engine.ReduceMean(squared, Enumerable.Range(0, squared.Shape.Length).ToArray(), keepDims: false);
    }

    /// <inheritdoc />
    /// <remarks>Characters <paramref name="input"/> and their mel spectrogram <paramref name="expectedOutput"/>
    /// <c>[1, frames, mel]</c>: the attention aligns them, so the pair is the whole supervision.</remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        Guard.NotNull(expectedOutput);
        if (!HasPaperLayers)
        {
            TrainWithTape(input, expectedOutput, _optimizer);
            return;
        }
        var target = MelRows(expectedOutput);
        if (target.Shape[0] == 0)
            throw new ArgumentException("expectedOutput has no mel frames.", nameof(expectedOutput));
        TrainWithCustomObjective(input, target, Objective, _optimizer);
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        Train(sample.Tokens, DeriveAcousticTargets(sample).Mel);
        return LastLoss ?? NumOps.Zero;
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
        => EvaluateObjective(sample.Tokens, DeriveAcousticTargets(sample).Mel);

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <remarks>Synthesis stops at the predicted stop token, so a prediction need not match a target frame for frame; the
    /// teacher-forced objective is what training lowers.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target) => EvaluateObjective(input, target);

    private T EvaluateObjective(Tensor<T> characters, Tensor<T> mel)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers)
            throw new NotSupportedException("The paper objective needs the paper's layers; this model was built from caller-supplied layers.");
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return Objective(characters, MelRows(mel))[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Tacotron2-Native" : "Tacotron2-ONNX",
            Description = "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions (Shen et al., 2018)",
            FeatureCount = _options.EmbeddingDim,
            Complexity = _options.NumEncoderLayers + _options.PostnetLayers,
        };
        m.AdditionalInfo["Architecture"] = "Tacotron2";
        m.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        m.AdditionalInfo["MelChannels"] = _options.MelChannels.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(Tacotron2<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
