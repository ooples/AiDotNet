using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// The training and synthesis plumbing of vocoders trained by one objective on aligned (mel spectrogram, waveform)
/// segments — the diffusion vocoders (DiffWave, WaveGrad, PriorGrad, FreGrad), the flow WaveGlow and the
/// autoregressive WaveNet and WaveRNN: a model supplies its network, input features, segment length, objective and
/// synthesis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>Training takes a frame-aligned segment of <see cref="SegmentSize"/> samples (rounded down to whole frames)
/// from a random frame, and one optimizer step on <see cref="TrainingObjective"/>. The random draws of a step (noise
/// levels, noise, dropout of the objective) come from one seed per step, so the objective is a deterministic function
/// of the parameters within the step. Synthesis runs with the model's sampling seed, so it is repeatable.</para>
/// <para><b>For Beginners:</b> This is the shared machinery: cut a matching piece of spectrogram and audio, compute the
/// model's loss on it, update the weights; to make speech, run the model's own generation procedure.</para>
/// </remarks>
public abstract partial class SegmentVocoderBase<T> : VocoderBase<T>, ITrainingObjectiveProvider<T>
{
    private readonly VocoderOptions _vocoder;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _suppliedOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _paperOptimizer;
    private readonly bool _useNativeMode;
    private bool _disposed;
    private readonly Random _trainingRandom;
    private readonly int _samplingSeed;
    private readonly List<LayerBase<T>> _networkLayers = new();

    /// <summary>Creates a native (trainable) vocoder.</summary>
    protected SegmentVocoderBase(NeuralNetworkArchitecture<T> architecture, VocoderOptions options,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer, int samplingSeed)
        : base(architecture)
    {
        _vocoder = options ?? throw new ArgumentNullException(nameof(options));
        // The clone plan replays the constructor from members named after its parameters: the options live in the base's
        // Options, as NeuralNetworkBase asks of every derived model.
        Options = _vocoder;
        _useNativeMode = true;
        _suppliedOptimizer = optimizer;
        _samplingSeed = samplingSeed;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(samplingSeed);
        ApplyAudioSettings();
        InitializeLayers();
    }

    /// <summary>Creates a vocoder that runs an exported ONNX graph.</summary>
    protected SegmentVocoderBase(NeuralNetworkArchitecture<T> architecture, string modelPath, VocoderOptions options, int samplingSeed)
        : base(architecture)
    {
        _vocoder = options ?? throw new ArgumentNullException(nameof(options));
        // The clone plan replays the constructor from members named after its parameters: the options live in the base's
        // Options, as NeuralNetworkBase asks of every derived model.
        Options = _vocoder;
        _useNativeMode = false;
        _samplingSeed = samplingSeed;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(samplingSeed);
        ApplyAudioSettings();
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _vocoder.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _vocoder.OnnxOptions);
        InitializeLayers();
    }

    private void ApplyAudioSettings()
    {
        base.SampleRate = _vocoder.SampleRate;
        base.MelChannels = _vocoder.MelChannels;
        base.HopSize = _vocoder.HopSize;
    }

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _vocoder;

    /// <summary>The options, available to the hooks the base constructor calls before a derived constructor runs.</summary>
    protected VocoderOptions VocoderSettings => _vocoder;

    /// <summary>Gets whether the paper's layers were built.</summary>
    protected bool HasPaperLayers => _networkLayers.Count > 0;

    /// <inheritdoc />
    public override IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => WaveformUpsampleContract(inputRank);

    // ---------------------------------------------------------------- hooks

    /// <summary>Builds the network, returning its layers.</summary>
    protected abstract IReadOnlyList<LayerBase<T>> CreateNetwork();

    /// <summary>The network's input features of a waveform <c>[samples]</c>, <c>[1, mel, frames]</c>; no gradient.</summary>
    protected abstract Tensor<T> ComputeInputMel(Tensor<T> audio);

    /// <summary>The training segment in samples.</summary>
    protected abstract int SegmentSize { get; }

    /// <summary>The training loss (a scalar on the gradient tape) of the mel spectrogram <paramref name="mel"/>
    /// <c>[1, mel, frames]</c> and its waveform <paramref name="audio"/> <c>[frames · hop]</c>, drawing any randomness
    /// from <paramref name="random"/>.</summary>
    protected abstract Tensor<T> TrainingObjective(Tensor<T> mel, Tensor<T> audio, Random random);

    /// <summary>The waveform <c>[1, 1, frames · hop]</c> of a mel spectrogram <c>[1, mel, frames]</c>, drawing any
    /// randomness from <paramref name="random"/>; called without a gradient tape.</summary>
    protected abstract Tensor<T> Synthesize(Tensor<T> mel, Random random);

    /// <summary>The number of seeded draws <see cref="ITrainingObjectiveProvider{T}.EvaluateTrainingObjective"/>
    /// averages (8: one draw of a stochastic objective is too noisy to compare across steps; 1 for a deterministic
    /// one).</summary>
    protected virtual int EvaluationDraws => 8;

    /// <summary>The global gradient-norm clip of a training step, 0 for none.</summary>
    protected virtual double GradientClipNorm => 0.0;

    /// <summary>Called after every optimizer step with the number of steps taken so far (WaveRNN's weight pruning).</summary>
    protected virtual void AfterTrainingStep(int step)
    {
    }

    /// <summary>The optimizer steps taken so far.</summary>
    protected int TrainingSteps { get; private set; }

    /// <summary>The optimizer as the paper configures it.</summary>
    protected abstract IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer();

    // ---------------------------------------------------------------- layers

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }
        _networkLayers.AddRange(CreateNetwork());
        AddEncoderDecoderLayers(_networkLayers.Cast<ILayer<T>>().ToList(), Array.Empty<ILayer<T>>());
    }

    // ---------------------------------------------------------------- helpers

    /// <summary>Standard-normal noise of <paramref name="shape"/>.</summary>
    protected Tensor<T> Gaussian(int[] shape, Random random)
    {
        var t = new Tensor<T>(shape);
        for (int i = 0; i < t.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            t[i] = NumOps.FromDouble(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
        }
        return t;
    }

    /// <summary>The mean of every element, as a scalar tensor.</summary>
    protected Tensor<T> Mean(Tensor<T> x) => Engine.ReduceMean(x, Enumerable.Range(0, x.Rank).ToArray(), keepDims: false);

    /// <summary>The log-softmax over the class axis of logits <c>[1, classes, T]</c>, with the per-step maximum held
    /// constant for stability.</summary>
    protected Tensor<T> LogSoftmaxOverClasses(Tensor<T> logits)
    {
        Tensor<T> max;
        using (new NoGradScope<T>())
        {
            var m = Engine.ReduceMax(logits, new[] { 1 }, keepDims: true);
            max = new Tensor<T>(m._shape, m.ToVector());
        }
        var shifted = Engine.TensorSubtract(logits, Engine.TensorTile(max, new[] { 1, logits.Shape[1], 1 }));
        var logSum = Engine.TensorLog(Engine.ReduceSum(Engine.TensorExp(shifted), new[] { 1 }, keepDims: true));
        return Engine.TensorSubtract(shifted, Engine.TensorTile(logSum, new[] { 1, logits.Shape[1], 1 }));
    }

    /// <summary>The mean cross-entropy of logits <c>[1, classes, T]</c> against class indices.</summary>
    protected Tensor<T> ClassCrossEntropy(Tensor<T> logits, int[] targets)
    {
        var oneHot = new Tensor<T>(logits._shape);
        for (int t = 0; t < targets.Length; t++) oneHot[0, targets[t], t] = NumOps.One;
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorMultiply(oneHot, LogSoftmaxOverClasses(logits)), new[] { 0, 1, 2 }, keepDims: false),
            NumOps.FromDouble(-1.0 / targets.Length));
    }

    /// <summary>A class drawn from the softmax of logits column <c>[1, classes, 1]</c>.</summary>
    protected int SampleClass(Tensor<T> logits, Random random)
    {
        int classes = logits.Shape[1];
        var p = new double[classes];
        double max = double.NegativeInfinity, sum = 0;
        for (int c = 0; c < classes; c++) max = Math.Max(max, NumOps.ToDouble(logits[0, c, 0]));
        for (int c = 0; c < classes; c++) sum += p[c] = Math.Exp(NumOps.ToDouble(logits[0, c, 0]) - max);
        double u = random.NextDouble() * sum;
        for (int c = 0; c < classes; c++)
        {
            u -= p[c];
            if (u <= 0) return c;
        }
        return classes - 1;
    }

    /// <summary>The tensor as a vector.</summary>
    protected Tensor<T> Flat(Tensor<T> x) => Engine.Reshape(x, new[] { x.Length });

    /// <summary>The input as <c>[1, mel, frames]</c>.</summary>
    protected Tensor<T> MelInput(Tensor<T> input)
    {
        var mel = input.Rank == 2 ? Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1] }) : input;
        if (mel.Rank != 3 || mel.Shape[1] != _vocoder.MelChannels)
            throw new ArgumentException($"Expected a mel spectrogram [1, {_vocoder.MelChannels}, frames], got [{string.Join(", ", input.Shape)}].", nameof(input));
        return mel;
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    public override Tensor<T> MelToWaveform(Tensor<T> melSpectrogram)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(melSpectrogram);
        return Predict(melSpectrogram);
    }

    /// <summary>The network's input features for <paramref name="audio"/> <c>[samples]</c>, <c>[1, mel, frames]</c>.</summary>
    public Tensor<T> ComputeMel(Tensor<T> audio)
    {
        using var _ = new NoGradScope<T>();
        return ComputeInputMel(Flat(audio));
    }

    /// <inheritdoc />
    protected override Tensor<T> PreprocessText(string text)
        => throw new NotSupportedException($"{GetType().Name} is a vocoder; it converts mel spectrograms (MelToWaveform), not text.");

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    /// <inheritdoc />
    /// <remarks>The model's synthesis, with randomness seeded by the model's sampling seed.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        SetTrainingMode(false);
        if (!HasPaperLayers)
        {
            var c = input;
            foreach (var l in Layers) c = l.Forward(c);
            return c;
        }
        using var _ = new NoGradScope<T>();
        return Synthesize(MelInput(input), AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_samplingSeed));
    }

    // ---------------------------------------------------------------- training

    /// <inheritdoc />
    /// <remarks>A mel spectrogram <paramref name="input"/> <c>[1, mel, frames]</c> aligned with the audio
    /// <paramref name="expectedOutput"/> trains on a frame-aligned segment of both.</remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers)
        {
            TrainWithTape(input, expectedOutput, _suppliedOptimizer);
            return;
        }
        var (mel, audio) = AlignedSegment(MelInput(input), expectedOutput, _trainingRandom);
        TrainStep(mel, audio);
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on recordings; set Audio.", nameof(sample));
        var flat = Flat(audio);
        var (mel, segment) = AlignedSegment(ComputeMel(flat), flat, _trainingRandom);
        TrainStep(mel, segment);
        return LastLoss ?? NumOps.Zero;
    }

    // A segment of SegmentSize samples (rounded down to whole frames) and its frames, from a random frame on.
    private (Tensor<T> Mel, Tensor<T> Audio) AlignedSegment(Tensor<T> mel, Tensor<T> audio, Random random)
    {
        int hop = UpsampleFactor, frames = mel.Shape[2];
        int segmentFrames = Math.Max(1, Math.Min(frames, SegmentSize / hop));
        int start = frames > segmentFrames ? random.Next(0, frames - segmentFrames + 1) : 0;
        var melSegment = Engine.TensorSlice(mel, new[] { 0, 0, start }, new[] { 1, _vocoder.MelChannels, segmentFrames });
        return (melSegment, AudioOfFrames(audio, start, segmentFrames));
    }

    // Samples [start · hop, (start + frames) · hop) of the audio, zero-padded past its end.
    private Tensor<T> AudioOfFrames(Tensor<T> audio, int start, int frames)
    {
        var flat = Flat(audio);
        int hop = UpsampleFactor, samples = frames * hop, from = start * hop;
        if (from + samples <= flat.Length)
            return Engine.TensorSlice(flat, new[] { from }, new[] { samples });
        int available = Math.Max(0, flat.Length - from);
        var head = Engine.TensorSlice(flat, new[] { Math.Min(from, flat.Length) }, new[] { available });
        return Engine.TensorConcatenate(new[] { head, new Tensor<T>(new[] { samples - available }) }, 0);
    }

    private void TrainStep(Tensor<T> mel, Tensor<T> audio)
    {
        SetTrainingMode(true);
        MaxGradNorm = NumOps.FromDouble(GradientClipNorm);
        int seed = _trainingRandom.Next();
        var flat = Flat(audio);
        TrainWithCustomObjective(mel, audio,
            (_, _) => TrainingObjective(mel, flat, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed)), OptimizerFor());
        TrainingSteps++;
        AfterTrainingStep(TrainingSteps);
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <remarks>The training loss on the whole input in evaluation mode, averaged over <see cref="EvaluationDraws"/>
    /// draws seeded by the model's sampling seed.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        ThrowIfDisposed();
        var mel = MelInput(input);
        var audio = AudioOfFrames(target, 0, mel.Shape[2]);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_samplingSeed + 11);
            int draws = Math.Max(1, EvaluationDraws);
            double total = 0;
            for (int i = 0; i < draws; i++) total += NumOps.ToDouble(TrainingObjective(mel, audio, random)[0]);
            return NumOps.FromDouble(total / draws);
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> OptimizerFor()
        => _suppliedOptimizer ?? (_paperOptimizer ??= CreatePaperOptimizer());

    /// <inheritdoc />
    /// <remarks>The paper optimizer reads the options when it is built, so restored options rebuild it.</remarks>
    protected override void OnMutableConstructorConfigurationRestored()
    {
        base.OnMutableConstructorConfigurationRestored();
        _paperOptimizer = null;
    }

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <summary>Throws when the model has been disposed.</summary>
    protected void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? GetType().Name);
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
