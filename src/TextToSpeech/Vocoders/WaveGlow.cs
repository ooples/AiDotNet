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
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, ReferenceBatchSize = 24,
                Source = "Prenger et al. 2019, Sec. 3.1: Adam with a step size of 1e-4 and a batch size of 24, 580k iterations (5e-5 after a plateau).")]
public partial class WaveGlow<T> : SegmentVocoderBase<T>
{
    private WaveGlowFlow<T>? _flow;
    private TacotronSpectrogram? _features;

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
        if (o.HopSize % o.GroupSize != 0)
            throw new ArgumentException($"The hop ({o.HopSize}) must be a multiple of the group size ({o.GroupSize}).");
        _flow = new WaveGlowFlow<T>(Engine, o.MelChannels, o.HopSize, o.UpsampleKernel, o.GroupSize, o.NumFlows, o.EarlyOutputEvery,
            o.EarlyOutputChannels, o.NumWaveNetLayers, o.ResidualChannels, o.GateChannels, o.SkipChannels, o.KernelSize);
        _features = new TacotronSpectrogram(o.SampleRate, o.FftSize, o.HopSize, o.WindowSize, o.MelChannels, o.MelMinFrequency, o.MelMaxFrequency);
        return _flow.Layers;
    }

    /// <inheritdoc />
    /// <remarks>Tacotron 2's log-mel spectrogram (reference <c>mel2samp.py</c>).</remarks>
    protected override Tensor<T> ComputeInputMel(Tensor<T> audio)
    {
        var samples = new double[audio.Length];
        for (int i = 0; i < samples.Length; i++) samples[i] = NumOps.ToDouble(audio[i]);
        var rows = _features!.LogMel(samples);
        int frames = rows.GetLength(0), bands = rows.GetLength(1);
        var mel = new Tensor<T>(new[] { 1, bands, frames });
        for (int f = 0; f < frames; f++)
            for (int m = 0; m < bands; m++) mel[0, m, f] = NumOps.FromDouble(rows[f, m]);
        return mel;
    }

    /// <summary>The latent z <c>[1, group, samples / group]</c> of a waveform <c>[samples]</c> given its mel spectrogram
    /// <c>[1, mel, frames]</c>, and the log-determinant of the flow.</summary>
    public (Tensor<T> Z, Tensor<T> LogDeterminant) Encode(Tensor<T> mel, Tensor<T> audio)
    {
        mel = MelInput(mel);
        var flat = Flat(audio);
        return _flow!.Forward(_flow.Squeeze(flat), _flow.Condition(mel, flat.Length));
    }

    /// <summary>The waveform <c>[1, 1, samples]</c> of a latent z <c>[1, group, samples / group]</c> (laid out as
    /// <see cref="Encode"/> returns it) given the mel spectrogram: the exact inverse of <see cref="Encode"/>.</summary>
    public Tensor<T> Decode(Tensor<T> mel, Tensor<T> z)
    {
        mel = MelInput(mel);
        var o = PaperOptions;
        int t = z.Shape[2], remaining = _flow!.RemainingChannels;
        // Encode emits the early outputs first, in order, then the remaining channels; the inverse consumes them in
        // reverse.
        var blocks = new Stack<Tensor<T>>();
        for (int from = 0; from < o.GroupSize - remaining; from += o.EarlyOutputChannels)
            blocks.Push(Engine.TensorSlice(z, new[] { 0, from, 0 }, new[] { 1, o.EarlyOutputChannels, t }));
        blocks.Push(Engine.TensorSlice(z, new[] { 0, o.GroupSize - remaining, 0 }, new[] { 1, remaining, t }));
        var x = _flow.Inverse(_ => blocks.Pop(), _flow.Condition(mel, t * o.GroupSize));
        return _flow.Unsqueeze(x);
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
        var condition = _flow!.Condition(mel, samples);
        var x = _flow.Inverse(shape => Engine.TensorMultiplyScalar(Gaussian(shape, random), NumOps.FromDouble(sigma)), condition);
        return _flow.Unsqueeze(x);
    }

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
