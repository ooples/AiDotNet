using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// WaveNet: a generative model for raw audio — an autoregressive stack of dilated causal convolutions that predicts a
/// softmax over μ-law-quantized samples, here locally conditioned on the mel spectrogram as a vocoder.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "WaveNet: A Generative Model for Raw Audio" (van den Oord et al., 2016) and
/// r9y9/wavenet_vocoder for what the paper leaves unstated.</para>
/// <para>
/// Training maximizes <c>Σ_t log p(x_t | x_1..x_{t−1}, y)</c> (Eq. 1, §2.5) with teacher forcing: the network reads the
/// one-hot μ-law classes of the previous samples and the upsampled mel y and predicts a softmax over the next sample's
/// class. Synthesis samples one class at a time from that softmax, feeding each back as the next input; each causal
/// convolution keeps the inputs it needs, so a step costs one column per layer.
/// </para>
/// <para><b>For Beginners:</b> WaveNet writes audio one sample at a time, each time choosing among 256 loudness levels
/// with probabilities computed from everything it has written so far and the spectrogram.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "WaveNet: A Generative Model for Raw Audio",
    "https://arxiv.org/abs/1609.03499",
    Year = 2016,
    Authors = "van den Oord et al."
)]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-3, Epsilon = 1e-8, ReferenceBatchSize = 8,
                Source = "The paper states no optimizer; r9y9/wavenet_vocoder mulaw256_wavenet.json: Adam at 1e-3 (eps 1e-8) halved every 200k steps, batch 8.")]
public partial class WaveNet<T> : SegmentVocoderBase<T>
{
    private WaveNetNetwork<T>? _network;
    private CenteredLogMel<T>? _features;

    /// <summary>Creates a WaveNet that runs an exported ONNX graph.</summary>
    public WaveNet(NeuralNetworkArchitecture<T> architecture, string modelPath, WaveNetOptions? options = null)
        : base(architecture, modelPath, options ?? new WaveNetOptions(), options?.SamplingSeed ?? 0)
    {
    }

    /// <summary>Creates a trainable WaveNet.</summary>
    public WaveNet(NeuralNetworkArchitecture<T> architecture, WaveNetOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new WaveNetOptions(), optimizer, options?.SamplingSeed ?? 0)
    {
    }

    private WaveNetOptions PaperOptions => (WaveNetOptions)VocoderSettings;

    /// <inheritdoc />
    public override int UpsampleFactor => PaperOptions.UpsampleScales.Aggregate(1, (a, b) => a * b);

    /// <inheritdoc />
    protected override int SegmentSize => PaperOptions.SegmentSamples;

    /// <inheritdoc />
    /// <remarks>The teacher-forced cross-entropy has no random draws.</remarks>
    protected override int EvaluationDraws => 1;

    /// <summary>The receptive field of the network in samples.</summary>
    public int ReceptiveField => _network?.ReceptiveField ?? 0;

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateNetwork()
    {
        var o = PaperOptions;
        if (UpsampleFactor != o.HopSize)
            throw new ArgumentException($"The upsampling scales ({string.Join("x", o.UpsampleScales)}) must multiply to the hop ({o.HopSize}).");
        _network = new WaveNetNetwork<T>(Engine, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed + 7),
            o.MuLawLevels, o.MelChannels, o.ResidualChannels, o.GateChannels, o.SkipChannels, o.NumDilatedLayers, o.DilationCycle,
            o.KernelSize, o.UpsampleScales);
        _features = new CenteredLogMel<T>(Engine, o.SampleRate, o.FftSize, o.HopSize, o.WindowSize, o.MelChannels, o.MelMinFrequency,
            o.MelMaxFrequency, 1e-10);
        return _network.Layers;
    }

    /// <inheritdoc />
    /// <remarks>The reference's <c>logmelspectrogram</c>: <c>log10(max(mel, 1e-10))</c> of a centred librosa STFT.</remarks>
    protected override Tensor<T> ComputeInputMel(Tensor<T> audio) => _features!.Forward(audio);

    // The upsampled condition [1, mel, samples].
    private Tensor<T> Condition(Tensor<T> mel, int samples)
    {
        var up = _network!.Upsample(mel);
        return Engine.TensorSlice(up, new[] { 0, 0, 0 }, new[] { 1, mel.Shape[1], samples });
    }

    private Tensor<T> OneHot(int[] classes)
    {
        int levels = PaperOptions.MuLawLevels;
        var t = new Tensor<T>(new[] { 1, levels, classes.Length });
        for (int i = 0; i < classes.Length; i++) t[0, classes[i], i] = NumOps.One;
        return t;
    }

    /// <summary>The μ-law classes of a waveform <c>[samples]</c>.</summary>
    public int[] Quantize(Tensor<T> audio)
    {
        int levels = PaperOptions.MuLawLevels;
        var q = new int[audio.Length];
        for (int i = 0; i < q.Length; i++) q[i] = MuLaw.Encode(NumOps.ToDouble(audio[i]), levels);
        return q;
    }

    /// <summary>The log-probabilities <c>[1, classes, samples]</c> of every sample's class given the previous samples
    /// of <paramref name="audio"/> <c>[samples]</c> and the mel spectrogram <c>[1, mel, frames]</c> (teacher forcing;
    /// the sample before the first is silence).</summary>
    public Tensor<T> LogProbabilities(Tensor<T> mel, Tensor<T> audio)
    {
        mel = MelInput(mel);
        var q = Quantize(audio);
        var previous = new int[q.Length];
        previous[0] = MuLaw.Encode(0, PaperOptions.MuLawLevels);
        for (int t = 1; t < q.Length; t++) previous[t] = q[t - 1];
        return LogSoftmaxOverClasses(_network!.Forward(OneHot(previous), Condition(mel, q.Length)));
    }

    /// <inheritdoc />
    /// <remarks>The mean cross-entropy of every sample's μ-law class (§2.2).</remarks>
    protected override Tensor<T> TrainingObjective(Tensor<T> mel, Tensor<T> audio, Random random)
    {
        var logp = LogProbabilities(mel, audio);
        var target = OneHot(Quantize(audio));
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorMultiply(target, logp), new[] { 0, 1, 2 }, keepDims: false),
            NumOps.FromDouble(-1.0 / audio.Length));
    }

    /// <inheritdoc />
    /// <remarks>Ancestral sampling of one μ-law class per sample from the softmax, starting after silence.</remarks>
    protected override Tensor<T> Synthesize(Tensor<T> mel, Random random)
    {
        var o = PaperOptions;
        int samples = mel.Shape[2] * UpsampleFactor, levels = o.MuLawLevels;
        var condition = Condition(mel, samples);
        var state = _network!.NewState();
        var wave = new Tensor<T>(new[] { 1, 1, samples });
        int previous = MuLaw.Encode(0, levels);
        for (int t = 0; t < samples; t++)
        {
            var logits = _network.Step(state, OneHot(new[] { previous }),
                Engine.TensorSlice(condition, new[] { 0, 0, t }, new[] { 1, mel.Shape[1], 1 }));
            int chosen = SampleClass(logits, random);
            wave[0, 0, t] = NumOps.FromDouble(MuLaw.Decode(chosen, levels));
            previous = chosen;
        }
        return wave;
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer()
    {
        var o = PaperOptions;
        return PaperOptimizerFactory.VerifyHandBuilt(this, new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = o.LearningRate,
                Epsilon = 1e-8,
                UseAdaptiveBetas = false,
                LearningRateScheduler = new AiDotNet.LearningRateSchedulers.StepLRScheduler(o.LearningRate,
                    Math.Max(1, o.LearningRateHalvingSteps), 0.5),
                SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
            }));
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = PaperOptions;
        var m = new ModelMetadata<T>
        {
            Name = IsOnnxMode ? "WaveNet-ONNX" : "WaveNet-Native",
            Description = "WaveNet: A Generative Model for Raw Audio (van den Oord et al., 2016)",
            FeatureCount = o.MelChannels,
            Complexity = o.NumDilatedLayers,
        };
        m.AdditionalInfo["Architecture"] = "WaveNet";
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        return m;
    }
}
