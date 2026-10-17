using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// DiffWave: a versatile diffusion model for audio synthesis — a non-autoregressive noise predictor built from
/// bidirectional dilated convolutions, conditioned on the mel spectrogram, sampled by the reverse diffusion process.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "DiffWave: A Versatile Diffusion Model for Audio Synthesis" (Kong et al., ICLR 2021) and
/// lmnt-com/diffwave for what the paper leaves unstated.</para>
/// <para>
/// Training (Algorithm 1): a step t uniform over the schedule, <c>x_t = √ᾱ_t x₀ + √(1 − ᾱ_t) ε</c>, and
/// <c>‖ε − ε_θ(x_t, t, mel)‖²</c>. Synthesis (Algorithm 2) runs the reverse process over the training schedule, or over
/// the six-step fast schedule (App. B, Algorithm 3), whose steps are aligned to fractional training steps by their noise
/// levels with the step embedding interpolated between neighbours.
/// </para>
/// <para><b>For Beginners:</b> DiffWave learns to remove noise from audio a little at a time; to synthesize, it starts
/// from noise and removes it step by step, guided by the spectrogram.</para>
/// </remarks>
/// <example>
/// <code>
/// // A vocoder: mel spectrogram in, waveform out.
/// var vocoder = new DiffWave&lt;float&gt;(architecture);
/// var speech = vocoder.MelToWaveform(mel);
/// // Unconditional and class-conditional generation (§5.2, §5.3).
/// var digits = new DiffWave&lt;float&gt;(architecture, DiffWaveOptions.ClassConditional(numClasses: 10));
/// digits.Train(new TtsTrainingSample&lt;float&gt; { Tokens = none, Audio = clip, ClassLabel = 7 });
/// var seven = digits.Generate(classLabel: 7);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "DiffWave: A Versatile Diffusion Model for Audio Synthesis",
    "https://arxiv.org/abs/2009.09761",
    Year = 2021,
    Authors = "Kong et al."
)]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 2e-4, ReferenceBatchSize = 16,
                Source = "Kong et al. 2021, Sec. 5.1: Adam with a batch size of 16 and a learning rate of 2e-4, 1M steps.")]
public partial class DiffWave<T> : DiffusionVocoderBase<T>
{
    private DiffWaveNetwork<T>? _network;
    private CenteredLogMel<T>? _features;

    /// <summary>Creates a DiffWave that runs an exported ONNX graph.</summary>
    public DiffWave(NeuralNetworkArchitecture<T> architecture, string modelPath, DiffWaveOptions? options = null)
        : base(architecture, modelPath, options ?? new DiffWaveOptions(), options?.SamplingSeed ?? 0)
    {
    }

    /// <summary>Creates a trainable DiffWave.</summary>
    public DiffWave(NeuralNetworkArchitecture<T> architecture, DiffWaveOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new DiffWaveOptions(), optimizer, options?.SamplingSeed ?? 0)
    {
    }

    private DiffWaveOptions PaperOptions => (DiffWaveOptions)VocoderSettings;

    /// <inheritdoc />
    public override int UpsampleFactor => PaperOptions.UpsampleStrides.Aggregate(1, (a, b) => a * b);

    /// <inheritdoc />
    protected override int SegmentSize => PaperOptions.CropFrames * UpsampleFactor;

    /// <inheritdoc />
    protected override double[] TrainingBetas => PaperOptions.NoiseSchedule;

    /// <inheritdoc />
    protected override double[]? InferenceBetas => PaperOptions.UseFastSampling ? PaperOptions.InferenceNoiseSchedule : null;

    /// <inheritdoc />
    protected override bool ClampEachStep => true;

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateNetwork()
    {
        var o = PaperOptions;
        if (UpsampleFactor != o.HopSize)
            throw new ArgumentException($"The upsampler's strides ({string.Join("x", o.UpsampleStrides)}) must multiply to the hop ({o.HopSize}).");
        if (o.Conditioner == DiffWaveConditioner.ClassLabel && o.NumClasses <= 0)
            throw new ArgumentException("A class-conditional DiffWave needs NumClasses (the dataset's label count).");
        if (o.Conditioner != DiffWaveConditioner.MelSpectrogram && o.UtteranceSamples <= 0)
            throw new ArgumentException("A DiffWave without a spectrogram needs UtteranceSamples, the length it generates.");
        _network = new DiffWaveNetwork<T>(Engine, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed + 7),
            o.MelChannels, o.ResChannels, o.NumResLayers, o.DilationCycle, o.NoiseSchedule.Length, o.UpsampleStrides,
            melConditioned: o.Conditioner == DiffWaveConditioner.MelSpectrogram,
            labelClasses: o.Conditioner == DiffWaveConditioner.ClassLabel ? o.NumClasses : 0);
        _features = new CenteredLogMel<T>(Engine, o.SampleRate, o.FftSize, o.HopSize, o.WindowSize, o.MelChannels, o.MelMinFrequency,
            o.SampleRate / 2.0, 1e-5, htkScale: true, normalizedWindow: true);
        return _network.Layers;
    }

    /// <inheritdoc />
    protected override Tensor<T> Denoise(Tensor<T> noisy, double level, Tensor<T> mel)
    {
        var network = _network ?? throw new InvalidOperationException("The DiffWave network has not been built.");
        return PaperOptions.Conditioner switch
        {
            DiffWaveConditioner.MelSpectrogram => network.Forward(noisy, level, mel),
            DiffWaveConditioner.Unconditional => network.Forward(noisy, level, (int?)null),
            _ => network.Forward(noisy, level, LabelOf(mel)),
        };
    }

    // A spectrogram-free condition is a one-element tensor holding the class label (ignored unconditionally).
    private int LabelOf(Tensor<T> condition)
    {
        double value = NumOps.ToDouble(condition[0]);
        int label = (int)Math.Round(value);
        if (Math.Abs(value - label) > 1e-9)
            throw new ArgumentException($"A class label must be an integer, got {value}.");
        return label;
    }

    /// <inheritdoc />
    /// <remarks>Without a spectrogram the input is the class label (a one-element tensor); unconditionally it is not
    /// read.</remarks>
    protected override Tensor<T> ConditionInput(Tensor<T> input)
    {
        if (PaperOptions.Conditioner == DiffWaveConditioner.MelSpectrogram) return MelInput(input);
        var condition = new Tensor<T>(new[] { 1 });
        if (PaperOptions.Conditioner == DiffWaveConditioner.ClassLabel)
        {
            if (input.Length != 1)
                throw new ArgumentException($"A class-conditional DiffWave takes its class label as a one-element tensor, got [{string.Join(", ", input.Shape)}].", nameof(input));
            condition[0] = input[0];
            LabelOf(condition);
        }
        return condition;
    }

    /// <inheritdoc />
    /// <remarks>Without a spectrogram the paper trains on full utterances, so the whole recording is used.</remarks>
    protected override (Tensor<T> Condition, Tensor<T> Audio) TrainingPair(Tensor<T> condition, Tensor<T> audio, Random random)
        => PaperOptions.Conditioner == DiffWaveConditioner.MelSpectrogram ? base.TrainingPair(condition, audio, random) : (condition, Flat(audio));

    /// <inheritdoc />
    /// <remarks>A class-conditional sample takes its label from <see cref="TtsTrainingSample{T}.ClassLabel"/>.</remarks>
    protected override Tensor<T> SampleCondition(TtsTrainingSample<T> sample, Tensor<T> audio)
    {
        switch (PaperOptions.Conditioner)
        {
            case DiffWaveConditioner.MelSpectrogram:
                return base.SampleCondition(sample, audio);
            case DiffWaveConditioner.ClassLabel:
                var label = new Tensor<T>(new[] { 1 });
                label[0] = NumOps.FromDouble(sample.ClassLabel
                    ?? throw new ArgumentException("A class-conditional DiffWave trains on labelled recordings; set ClassLabel.", nameof(sample)));
                return ConditionInput(label);
            default:
                return new Tensor<T>(new[] { 1 });
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> EvaluationAudio(Tensor<T> condition, Tensor<T> target)
        => PaperOptions.Conditioner == DiffWaveConditioner.MelSpectrogram ? base.EvaluationAudio(condition, target) : target;

    /// <inheritdoc />
    protected override int SynthesisSamples(Tensor<T> mel)
        => PaperOptions.Conditioner == DiffWaveConditioner.MelSpectrogram ? base.SynthesisSamples(mel) : PaperOptions.UtteranceSamples;

    /// <summary>Generates an utterance of <see cref="DiffWaveOptions.UtteranceSamples"/> samples by the reverse process
    /// (Algorithm 2): unconditionally, or of class <paramref name="classLabel"/> for a class-conditional model.</summary>
    public Tensor<T> Generate(int? classLabel = null)
    {
        var conditioner = PaperOptions.Conditioner;
        if (conditioner == DiffWaveConditioner.MelSpectrogram)
            throw new InvalidOperationException("This DiffWave is a vocoder; convert a mel spectrogram with MelToWaveform.");
        if ((conditioner == DiffWaveConditioner.ClassLabel) != classLabel.HasValue)
            throw new ArgumentException(classLabel.HasValue ? "An unconditional DiffWave takes no class label." : "A class-conditional DiffWave needs a class label.", nameof(classLabel));
        var input = new Tensor<T>(new[] { 1 });
        input[0] = NumOps.FromDouble(classLabel ?? 0);
        return Predict(input);
    }

    /// <inheritdoc />
    /// <remarks><c>clamp((20 log10(max(mel, 1e-5)) − 20 + 100) / 100, 0, 1)</c> (reference <c>preprocess.py</c>).</remarks>
    protected override Tensor<T> ComputeInputMel(Tensor<T> audio) => DecibelUnitRange(_features!.Forward(audio));

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer()
        => PaperOptimizerFactory.VerifyHandBuilt(this, new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = PaperOptions.LearningRate,
                UseAdaptiveBetas = false,
            }));

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = PaperOptions;
        var m = new ModelMetadata<T>
        {
            Name = IsOnnxMode ? "DiffWave-ONNX" : "DiffWave-Native",
            Description = "DiffWave: A Versatile Diffusion Model for Audio Synthesis (Kong et al., 2021)",
            FeatureCount = o.MelChannels,
            Complexity = o.NumResLayers,
        };
        m.AdditionalInfo["Architecture"] = "DiffWave";
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        return m;
    }
}
