using AiDotNet.Helpers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// WaveGlow: a flow-based generative network for speech synthesis — Glow's invertible 1×1 convolutions and affine
/// couplings over groups of audio samples, with WaveNet-like coupling networks conditioned on the mel spectrogram,
/// trained only by maximizing the likelihood.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "WaveGlow: A Flow-based Generative Network for Speech Synthesis" (Prenger et al., ICASSP
/// 2019) and NVIDIA/waveglow for what the paper leaves unstated.</para>
/// <para>
/// Training maximizes <c>log p(x) = −z(x)ᵀz(x) / 2σ² + Σ log s + Σ log|det W|</c> (Eq. 12) of a clip, squeezed into groups
/// of 8 samples and passed through the steps of flow with early outputs (§2.3). Synthesis draws z with a smaller
/// σ (§2.4) and inverts every step.
/// </para>
/// <para><b>For Beginners:</b> WaveGlow learns a reversible mapping between speech and random noise; to make speech it
/// draws noise and runs the mapping backwards, guided by the spectrogram.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "WaveGlow: A Flow-based Generative Network for Speech Synthesis",
    "https://arxiv.org/abs/1811.00002",
    Year = 2019,
    Authors = "Prenger et al."
)]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, ReferenceBatchSize = 24,
                Source = "Prenger et al. 2019, Sec. 3.1: Adam with a step size of 1e-4 and a batch size of 24, 580k iterations (5e-5 after a plateau). "
                    + "NVIDIA/waveglow train.py uses torch.optim.Adam's fixed default betas (0.9, 0.999) and epsilon 1e-8.")]
public partial class WaveGlow<T> : SegmentVocoderBase<T>
{
    private WaveGlowFlow<T>? _flow;
    private WaveGlowFlow<T> RequiredFlow => _flow ?? throw new InvalidOperationException("The network has not been created.");
    private TacotronSpectrogram? _features;
    private TacotronSpectrogram RequiredFeatures => _features ?? throw new InvalidOperationException("The network has not been created.");

    /// <summary>Creates a WaveGlow that runs an exported ONNX graph.</summary>
    public WaveGlow(NeuralNetworkArchitecture<T> architecture, string modelPath, WaveGlowOptions? options = null)
        : base(architecture, modelPath, options ?? new WaveGlowOptions(), options?.SamplingSeed ?? 0)
    {
    }

    /// <summary>Creates a trainable WaveGlow.</summary>
    public WaveGlow(NeuralNetworkArchitecture<T> architecture, WaveGlowOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new WaveGlowOptions(), optimizer, options?.SamplingSeed ?? 0)
    {
    }

    private WaveGlowOptions PaperOptions => (WaveGlowOptions)VocoderSettings;

    /// <inheritdoc />
    public override int UpsampleFactor => PaperOptions.HopSize;

    /// <inheritdoc />
    protected override int SegmentSize => PaperOptions.SegmentSamples;

    /// <inheritdoc />
    /// <remarks>The likelihood of a clip has no random draws.</remarks>
    protected override int EvaluationDraws => 1;

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateNetwork()
    {
        var o = PaperOptions;
        if (o.GroupSize < 2 || o.HopSize <= 0 || o.HopSize % o.GroupSize != 0)
            throw new ArgumentException($"The hop ({o.HopSize}) must be a multiple of the group size ({o.GroupSize}).");
        if (o.SegmentSamples < o.GroupSize)
            throw new ArgumentOutOfRangeException(nameof(o.SegmentSamples), "A recording segment must contain at least one audio group.");
        _flow = new WaveGlowFlow<T>(Engine, o.MelChannels, o.HopSize, o.UpsampleKernel, o.GroupSize, o.NumFlows, o.EarlyOutputEvery,
            o.EarlyOutputChannels, o.NumWaveNetLayers, o.ResidualChannels, o.GateChannels, o.SkipChannels, o.KernelSize);
        _features = new TacotronSpectrogram(o.SampleRate, o.FftSize, o.HopSize, o.WindowSize, o.MelChannels, o.MelMinFrequency, o.MelMaxFrequency);
        return _flow.Layers;
    }

    /// <inheritdoc />
    /// <remarks>Tacotron 2's log-mel spectrogram (reference <c>mel2samp.py</c>).</remarks>
    protected override Tensor<T> ComputeInputMel(Tensor<T> audio)
    {
        var rows = RequiredFeatures.LogMel(TensorCopyHelper.CastLogical<T, double>(audio).ToArray());
        int frames = rows.GetLength(0), bands = rows.GetLength(1);
        var flat = new double[rows.Length];
        Buffer.BlockCopy(rows, 0, flat, 0, rows.Length * sizeof(double));
        return Engine.Reshape(Engine.TensorTranspose(new Tensor<double>(flat, new[] { frames, bands })), new[] { 1, bands, frames })
            .Cast<T>();
    }

    /// <inheritdoc />
    protected override (Tensor<T> Condition, Tensor<T> Audio) RecordingPair(TtsTrainingSample<T> sample, Tensor<T> audio, Random random)
        => PrepareRecording(audio, random);

    /// <summary>NVIDIA mel2samp.py: crop or zero-pad exactly segment_length raw samples, then compute their mel features.
    /// The released 16,000-sample crop is intentionally not rounded down to a 256-sample hop boundary.</summary>
    internal (Tensor<T> Condition, Tensor<T> Audio) PrepareRecording(Tensor<T> recording, Random random)
    {
        var flat = Flat(recording);
        int count = PaperOptions.SegmentSamples;
        Tensor<T> segment;
        if (flat.Length >= count)
        {
            int start = random.Next(0, flat.Length - count + 1);
            segment = Engine.TensorSlice(flat, new[] { start }, new[] { count });
        }
        else
            segment = Engine.TensorConcatenate(new[] { flat, new Tensor<T>(new[] { count - flat.Length }) }, 0);
        return (ComputeMel(segment), segment);
    }

    /// <inheritdoc />
    /// <remarks>Caller-supplied mel/audio pairs retain their alignment. Bound the sample count, rather than rounding
    /// the requested segment to whole frames; raw recordings use the released preprocessing above.</remarks>
    protected override (Tensor<T> Condition, Tensor<T> Audio) TrainingPair(Tensor<T> condition, Tensor<T> audio, Random random)
        => PrepareAligned(condition, audio, random);

    internal (Tensor<T> Condition, Tensor<T> Audio) PrepareAligned(Tensor<T> condition, Tensor<T> audio, Random random)
    {
        var flat = Flat(audio);
        var o = PaperOptions;
        int frames = condition.Shape[2];
        if (frames < 1) throw new ArgumentException("At least one mel frame is required.", nameof(condition));
        int capacity = checked((frames - 1) * o.HopSize + o.UpsampleKernel);
        int count = Math.Min(o.SegmentSamples, Math.Min(flat.Length, capacity));
        if (count < o.GroupSize) throw new ArgumentException("The aligned pair must contain at least one complete audio group.", nameof(audio));
        int last = Math.Min(frames - 1, Math.Min((flat.Length - count) / o.HopSize, (capacity - count) / o.HopSize));
        int start = random.Next(0, last + 1);
        int melFrames = Math.Min(frames - start, count / o.HopSize + 1);
        var mel = Engine.TensorSlice(condition, new[] { 0, 0, start }, new[] { 1, o.MelChannels, melFrames });
        return (mel, Engine.TensorSlice(flat, new[] { start * o.HopSize }, new[] { count }));
    }

    /// <inheritdoc />
    protected override Tensor<T> EvaluationAudio(Tensor<T> condition, Tensor<T> target) => Flat(target);

    /// <summary>The latent z <c>[1, group, samples / group]</c> of a waveform <c>[samples]</c> given its mel spectrogram
    /// <c>[1, mel, frames]</c>, and the log-determinant of the flow.</summary>
    public (Tensor<T> Z, Tensor<T> LogDeterminant) Encode(Tensor<T> mel, Tensor<T> audio)
    {
        mel = MelInput(mel);
        var flat = Flat(audio);
        return RequiredFlow.Forward(RequiredFlow.Squeeze(flat), RequiredFlow.Condition(mel, flat.Length));
    }

    /// <summary>The waveform <c>[1, 1, samples]</c> of a latent z <c>[1, group, samples / group]</c> (laid out as
    /// <see cref="Encode"/> returns it) given the mel spectrogram: the exact inverse of <see cref="Encode"/>.</summary>
    public Tensor<T> Decode(Tensor<T> mel, Tensor<T> z)
    {
        mel = MelInput(mel);
        var o = PaperOptions;
        int t = z.Shape[2], remaining = RequiredFlow.RemainingChannels;
        // Encode emits the early outputs first, in order, then the remaining channels; the inverse consumes them in
        // reverse.
        var blocks = new Stack<Tensor<T>>();
        for (int from = 0; from < o.GroupSize - remaining; from += o.EarlyOutputChannels)
            blocks.Push(Engine.TensorSlice(z, new[] { 0, from, 0 }, new[] { 1, o.EarlyOutputChannels, t }));
        blocks.Push(Engine.TensorSlice(z, new[] { 0, o.GroupSize - remaining, 0 }, new[] { 1, remaining, t }));
        var x = RequiredFlow.Inverse(_ => blocks.Pop(), RequiredFlow.Condition(mel, t * o.GroupSize));
        return RequiredFlow.Unsqueeze(x);
    }

    /// <inheritdoc />
    /// <remarks>The negative log-likelihood per sample, <c>(Σz² / 2σ² − Σ log s − Σ log|det W|) / samples</c>, without
    /// the constant (reference <c>WaveGlowLoss</c>).</remarks>
    protected override Tensor<T> TrainingObjective(Tensor<T> mel, Tensor<T> audio, Random random)
    {
        var (z, logDet) = Encode(mel, audio);
        double sigma = PaperOptions.TrainingSigma;
        var energy = Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorMultiply(z, z), new[] { 0, 1, 2 }, keepDims: false),
            NumOps.FromDouble(1 / (2 * sigma * sigma)));
        return Engine.TensorMultiplyScalar(Engine.TensorSubtract(energy, logDet), NumOps.FromDouble(1.0 / z.Length));
    }

    /// <inheritdoc />
    /// <remarks>z ~ N(0, σ²) at the inference σ, inverted through every step (§2.4).</remarks>
    protected override Tensor<T> Synthesize(Tensor<T> mel, Random random)
    {
        double sigma = PaperOptions.InferenceSigma;
        int samples = mel.Shape[2] * UpsampleFactor;
        var condition = RequiredFlow.Condition(mel, samples);
        var x = RequiredFlow.Inverse(shape => Engine.TensorMultiplyScalar(Gaussian(shape, random), NumOps.FromDouble(sigma)), condition);
        return RequiredFlow.Unsqueeze(x);
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer()
    {
        var optimizer = PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this,
            learningRateOverride: PaperOptions.LearningRate, weightDecayOverride: PaperOptions.WeightDecay)
            ?? throw new InvalidOperationException("WaveGlow's native Adam recipe could not be created.");
        // Neither the paper nor the released train.py clips WaveGlow gradients.
        if (optimizer.GetOptions() is GradientBasedOptimizerOptions<T, Tensor<T>, Tensor<T>> settings)
            settings.EnableGradientClipping = false;
        return optimizer;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = PaperOptions;
        var m = new ModelMetadata<T>
        {
            Name = IsOnnxMode ? "WaveGlow-ONNX" : "WaveGlow-Native",
            Description = "WaveGlow: A Flow-based Generative Network for Speech Synthesis (Prenger et al., 2019)",
            FeatureCount = o.MelChannels,
            Complexity = o.NumFlows,
        };
        m.AdditionalInfo["Architecture"] = "WaveGlow";
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        return m;
    }
}
