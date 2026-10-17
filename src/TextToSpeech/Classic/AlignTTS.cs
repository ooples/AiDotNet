using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.ActivationFunctions;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>
/// AlignTTS: alignment-free non-autoregressive TTS using feed-forward transformer with mix density network alignment.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "AlignTTS: Efficient Feed-Forward Text-to-Speech Model without Explicit Alignment" (Zeng et al., 2020)</item></list></para>
/// <para><b>For Beginners:</b> AlignTTS is a non-autoregressive text-to-speech model that converts text input into speech audio output.</para>
/// <example>
/// <code>
/// // Create an AlignTTS model for alignment-free non-autoregressive speech synthesis
/// // with feed-forward transformer and mix density network alignment
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 200, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new AlignTTS&lt;double&gt;(architecture, "aligntts.onnx");
///
/// // Training mode with native layers
/// var trainModel = new AlignTTS&lt;double&gt;(architecture, new AlignTTSOptions());
/// </code>
/// </example>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "AlignTTS: Efficient Feed-Forward Text-to-Speech Model without Explicit Alignment",
    "https://arxiv.org/abs/2003.01950",
    Year = 2020,
    Authors = "Zeng et al."
)]
[PaperOptimizer(OptimizerKind.Adam, Beta1 = 0.9, Beta2 = 0.98, Epsilon = 1e-9,
                ReferenceBatchSize = 32, Schedule = LearningRateSchedulerType.Noam,
                Phase = TrainingPhase.PreTraining,
                Provenance = RecipeProvenance.DerivedFromCitedWork,
                Source = "Zeng et al. 2020, Sec. 4.1: Adam with beta1 0.9, beta2 0.98 and epsilon 1e-9, "
                        + "at a batch size of 16 samples on each of 2 GPUs, for 40K steps in the first "
                        + "two training stages. The schedule is not restated -- the paper adopts the one "
                        + "from its reference [18], Vaswani et al. 2017, which is the "
                        + "inverse-square-root schedule with warmup declared here as Noam; the "
                        + "provenance records that it comes from the cited work rather than from this "
                        + "paper.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4,
                Phase = TrainingPhase.FineTuning,
                Source = "Zeng et al. 2020, Sec. 4.1: fine-tuning the whole model uses a fixed learning "
                        + "rate of 1e-4 over 80K steps.")]
public partial class AlignTTS<T> : TtsModelBase<T>, IAcousticModel<T>, ITrainingObjectiveProvider<T>
{
    private readonly AlignTTSOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _fineTuningOptimizer;
    private bool _useNativeMode;
    private bool _disposed;

    // Separate networks the paper trains alongside the feed-forward Transformer (§2.2, §2.3). They are not part of
    // the sequential layer stack; registering them in ComponentLayers gives them training, counting, serialization
    // and cloning.
    private readonly List<LayerBase<T>> _durationPredictor = new();
    private readonly List<LayerBase<T>> _mixDensityNetwork = new();

    public override ModelOptions GetOptions() => _options;

    public AlignTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, AlignTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new AlignTTSOptions();
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

    public AlignTTS(
        NeuralNetworkArchitecture<T> architecture,
        AlignTTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new AlignTTSOptions();
        _useNativeMode = true;
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        InitializeLayers();
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _fineTuningOptimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this, phase: AiDotNet.Attributes.TrainingPhase.FineTuning)
            ?? _optimizer;
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public new int MelChannels => _options.MelChannels;
    public new int HopSize => _options.HopSize;
    public int FftSize => _options.FftSize;

    /// <summary>
    /// The paper's training phase that <c>Train</c> runs (Zeng et al. 2020, §3.2.2). Training starts with
    /// <see cref="AlignTTSTrainingPhase.Alignment"/>; advance through the phases in order.
    /// </summary>
    public AlignTTSTrainingPhase CurrentPhase { get; set; } = AlignTTSTrainingPhase.Alignment;

    /// <summary>Generates a mel spectrogram from text (AlignTTS is an acoustic model; pair it with a vocoder).</summary>
    public Tensor<T> TextToMel(string text) => Synthesize(text);

    /// <summary>
    /// Builds the feed-forward Transformer (character embedding and positions, character-side FFT blocks; positions,
    /// mel-side FFT blocks and the linear projection), the separate duration predictor (character embedding,
    /// FFT blocks, linear) and the mix density network (stacked linear + LayerNorm + ReLU + dropout, then a linear
    /// layer giving each character's Gaussian mean and log-variance over the mel channels).
    /// </summary>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }

        int k = _options.FftKernelSize;
        AddEncoderDecoderLayers(
            LayerHelper<T>.CreateDefaultFastSpeechEncoderLayers(
                _options.VocabSize, _options.EncoderDim, _options.HiddenDim, _options.NumEncoderLayers, _options.NumHeads,
                _options.FftFilterSize, k, k, _options.DropoutRate, _options.MaxTextLength),
            AlignTTSDecoderLayers(k));

        _durationPredictor.Add(new EmbeddingLayer<T>(_options.VocabSize, _options.DurationPredictorDim));
        _durationPredictor.Add(new PositionalEncodingLayer<T>(_options.MaxTextLength, _options.DurationPredictorDim));
        for (int i = 0; i < _options.DurationPredictorLayers; i++)
            _durationPredictor.Add(new FeedForwardTransformerBlock<T>(_options.DurationPredictorDim, _options.NumHeads,
                _options.DurationPredictorDim, k, k, _options.DropoutRate));
        _durationPredictor.Add(new DenseLayer<T>(1, new IdentityActivation<T>() as IActivationFunction<T>));

        for (int i = 0; i < _options.MixDensityHiddenLayers; i++)
        {
            _mixDensityNetwork.Add(new DenseLayer<T>(_options.MixDensityHiddenSize, new IdentityActivation<T>() as IActivationFunction<T>));
            _mixDensityNetwork.Add(new LayerNormalizationLayer<T>(_options.MixDensityHiddenSize));
            _mixDensityNetwork.Add(new ActivationLayer<T>(new ReLUActivation<T>() as IActivationFunction<T>));
            if (_options.DropoutRate > 0) _mixDensityNetwork.Add(new DropoutLayer<T>(_options.DropoutRate));
        }
        _mixDensityNetwork.Add(new DenseLayer<T>(2 * _options.MelChannels, new IdentityActivation<T>() as IActivationFunction<T>));

        ComponentLayers.AddRange(_durationPredictor);
        ComponentLayers.AddRange(_mixDensityNetwork);
    }

    private IEnumerable<ILayer<T>> AlignTTSDecoderLayers(int kernel)
    {
        yield return new PositionalEncodingLayer<T>(_options.MaxMelLength, _options.HiddenDim);
        for (int i = 0; i < _options.NumDecoderLayers; i++)
            yield return new FeedForwardTransformerBlock<T>(_options.HiddenDim, _options.NumHeads, _options.FftFilterSize,
                kernel, kernel, _options.DropoutRate);
        yield return new DenseLayer<T>(_options.MelChannels, new IdentityActivation<T>() as IActivationFunction<T>);
    }

    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    /// <summary>
    /// Inference (§3.3): the character-side blocks encode the text, the duration predictor sets each character's
    /// length, the length regulator expands, and the mel-side blocks produce the spectrogram.
    /// </summary>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        SetTrainingMode(false);
        var hidden = RunEncoder(input);
        var durations = PredictDurations(input);
        return RunDecoder(LengthRegulator.Expand(hidden, durations));
    }

    private int[] PredictDurations(Tensor<T> tokens)
    {
        var logDuration = RunDurationPredictor(tokens);
        var durations = new int[logDuration.Length];
        int total = 0;
        for (int i = 0; i < durations.Length; i++)
        {
            durations[i] = (int)Math.Max(0, Math.Round(Math.Exp(NumOps.ToDouble(logDuration[i])) - 1.0));
            total += durations[i];
        }
        if (total == 0)
            for (int i = 0; i < durations.Length; i++) durations[i] = 1;
        return durations;
    }

    private Tensor<T> RunDurationPredictor(Tensor<T> tokens)
    {
        var x = tokens;
        foreach (var layer in _durationPredictor) x = layer.Forward(x);
        return Engine.Reshape(x, new[] { tokens.Length });
    }

    /// <summary>log N(y_t | μ_s, diag σ²_s) for every character s and frame t, <c>[characters, frames]</c>.</summary>
    private Tensor<T> GaussianLogLikelihood(Tensor<T> characterHidden, Tensor<T> mel)
    {
        var x = characterHidden;
        foreach (var layer in _mixDensityNetwork) x = layer.Forward(x);
        int characters = characterHidden.Shape[0], channels = _options.MelChannels, frames = mel.Shape[0];
        var mean = Engine.TensorSlice(x, new[] { 0, 0 }, new[] { characters, channels });
        var logVariance = Engine.TensorSlice(x, new[] { 0, channels }, new[] { characters, channels });

        // Σ_d [log 2π + logσ²_d + (y_d - μ_d)² / σ²_d] for every (character, frame) pair, via
        // (y-μ)²/σ² = y²/σ² - 2yμ/σ² + μ²/σ², each term a matrix product over the channels.
        var precision = Engine.TensorExp(Engine.TensorNegate(logVariance));                  // [S, D]
        var melT = Engine.TensorTranspose(mel);                                               // [D, F]
        var melSq = Engine.TensorMultiply(melT, melT);
        var quadratic = Engine.TensorMatMul(precision, melSq);                               // [S, F]
        var cross = Engine.TensorMatMul(Engine.TensorMultiply(precision, mean), melT);       // [S, F]
        var constant = Engine.ReduceSum(Engine.TensorAdd(logVariance, Engine.TensorMultiply(precision, Engine.TensorMultiply(mean, mean))),
            new[] { 1 }, keepDims: true);                                                      // [S, 1]
        var sum = Engine.TensorAdd(Engine.TensorSubtract(quadratic, Engine.TensorMultiplyScalar(cross, NumOps.FromDouble(2))),
            Engine.TensorTile(constant, new[] { 1, frames }));
        var log2pi = NumOps.FromDouble(channels * Math.Log(2 * Math.PI));
        return Engine.TensorMultiplyScalar(Engine.TensorAddScalar(sum, log2pi), NumOps.FromDouble(-0.5));
    }

    private int[] ViterbiDurations(Tensor<T> logLikelihood)
    {
        int characters = logLikelihood.Shape[0], frames = logLikelihood.Shape[1];
        var scores = new double[characters, frames];
        for (int s = 0; s < characters; s++)
            for (int t = 0; t < frames; t++) scores[s, t] = NumOps.ToDouble(logLikelihood[s, t]);
        return MonotonicAlignment.MaximumPath(scores);
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        var optimizer = CurrentPhase is AlignTTSTrainingPhase.JointFineTuning or AlignTTSTrainingPhase.DurationPredictor
            ? _fineTuningOptimizer
            : _optimizer;
        TrainWithCustomObjective(input, expected, PhaseObjective, optimizer);
    }

    /// <summary>
    /// The current phase's objective (§3.2.2): (1) the alignment loss on the character-side blocks and MDN;
    /// (2) the mel MSE on the mel side with Viterbi durations from the MDN; (3) both, durations recomputed each
    /// step; (4) the log-duration MSE of the duration predictor against the MDN's durations.
    /// </summary>
    private Tensor<T> PhaseObjective(Tensor<T> tokens, Tensor<T> mel)
    {
        var melFrames = mel.Rank == 3 ? Engine.Reshape(mel, new[] { mel.Shape[1], mel.Shape[2] }) : mel;
        int frames = melFrames.Shape[0];
        var hidden = RunEncoder(tokens);
        var characterHidden = hidden.Rank == 3 ? Engine.Reshape(hidden, new[] { hidden.Shape[1], hidden.Shape[2] }) : hidden;

        Tensor<T> AlignmentLoss(Tensor<T> logLikelihood)
            => Engine.TensorMultiplyScalar(MonotonicAlignment.LogLikelihood(logLikelihood), NumOps.FromDouble(-1.0 / frames));

        switch (CurrentPhase)
        {
            case AlignTTSTrainingPhase.Alignment:
                return AlignmentLoss(GaussianLogLikelihood(characterHidden, melFrames));

            case AlignTTSTrainingPhase.Decoder:
            {
                int[] durations;
                using (new NoGradScope<T>())
                    durations = ViterbiDurations(GaussianLogLikelihood(characterHidden, melFrames));
                var expanded = LengthRegulator.Expand(characterHidden, durations);
                return MeanSquaredError(RunDecoder(expanded), melFrames);
            }

            case AlignTTSTrainingPhase.JointFineTuning:
            {
                var logLikelihood = GaussianLogLikelihood(characterHidden, melFrames);
                int[] durations;
                using (new NoGradScope<T>())
                    durations = ViterbiDurations(logLikelihood);
                var expanded = LengthRegulator.Expand(characterHidden, durations);
                return Engine.TensorAdd(MeanSquaredError(RunDecoder(expanded), melFrames), AlignmentLoss(logLikelihood));
            }

            default:
            {
                int[] durations;
                using (new NoGradScope<T>())
                    durations = ViterbiDurations(GaussianLogLikelihood(characterHidden, melFrames));
                var target = new Tensor<T>(new[] { durations.Length });
                for (int i = 0; i < durations.Length; i++) target[i] = NumOps.FromDouble(Math.Log(durations[i] + 1.0));
                return MeanSquaredError(RunDurationPredictor(tokens), target);
            }
        }
    }

    /// <inheritdoc />
    /// <remarks>Each phase updates only the parameters the paper trains in it (§3.2.2).</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        var encoder = Training.TapeTrainingStep<T>.CollectParameters(Layers.Take(EncoderLayerCount).ToList(), -1);
        var decoder = Training.TapeTrainingStep<T>.CollectParameters(Layers.Skip(EncoderLayerCount).ToList(), -1);
        var duration = Training.TapeTrainingStep<T>.CollectParameters(_durationPredictor.Cast<ILayer<T>>().ToList(), -1);
        var mdn = Training.TapeTrainingStep<T>.CollectParameters(_mixDensityNetwork.Cast<ILayer<T>>().ToList(), -1);
        IEnumerable<Tensor<T>> selected = CurrentPhase switch
        {
            AlignTTSTrainingPhase.Alignment => encoder.Concat(mdn),
            AlignTTSTrainingPhase.Decoder => decoder,
            AlignTTSTrainingPhase.JointFineTuning => encoder.Concat(decoder).Concat(mdn),
            _ => duration,
        };
        var keep = new HashSet<Tensor<T>>(selected, Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(keep.Contains).ToList();
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return PhaseObjective(input, target)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private Tensor<T> MeanSquaredError(Tensor<T> prediction, Tensor<T> target)
    {
        var diff = Engine.TensorSubtract(prediction, Engine.Reshape(target, prediction._shape));
        var sq = Engine.TensorMultiply(diff, diff);
        return Engine.ReduceMean(sq, Enumerable.Range(0, sq.Rank).ToArray(), keepDims: false);
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "AlignTTS-Native" : "AlignTTS-ONNX",
            Description =
                "AlignTTS: Efficient Feed-Forward TTS without Explicit Alignment (Zeng et al., 2020)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "AlignTTS";
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(AlignTTS<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}

/// <summary>The training phases of AlignTTS (Zeng et al. 2020, §3.2.2), run in order.</summary>
public enum AlignTTSTrainingPhase
{
    /// <summary>Train the mix density network and the character-side blocks with the alignment loss.</summary>
    Alignment = 1,

    /// <summary>Freeze the character side; train the mel side on mel MSE with the MDN's Viterbi durations.</summary>
    Decoder = 2,

    /// <summary>Fine-tune the feed-forward Transformer and the MDN together, durations recomputed every step.</summary>
    JointFineTuning = 3,

    /// <summary>Train the duration predictor on the final MDN durations (log-domain MSE).</summary>
    DurationPredictor = 4,
}
