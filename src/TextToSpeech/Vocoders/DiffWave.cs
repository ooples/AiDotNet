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
        _network = new DiffWaveNetwork<T>(Engine, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed + 7),
            o.MelChannels, o.ResChannels, o.NumResLayers, o.DilationCycle, o.NoiseSchedule.Length, o.UpsampleStrides);
        _features = new CenteredLogMel<T>(Engine, o.SampleRate, o.FftSize, o.HopSize, o.WindowSize, o.MelChannels, o.MelMinFrequency,
            o.SampleRate / 2.0, 1e-5, htkScale: true, normalizedWindow: true);
        return _network.Layers;
    }

    /// <inheritdoc />
    protected override Tensor<T> Denoise(Tensor<T> noisy, double level, Tensor<T> mel) => _network!.Forward(noisy, level, mel);

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
