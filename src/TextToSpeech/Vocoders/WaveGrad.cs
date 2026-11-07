using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// WaveGrad: estimating gradients for waveform generation — a diffusion vocoder whose noise predictor is conditioned on
/// the continuous noise level √ᾱ, built from GAN-TTS-style upsampling blocks over the mel spectrogram modulated by
/// FiLM from downsampling blocks over the noisy waveform.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "WaveGrad: Estimating Gradients for Waveform Generation" (Chen et al., ICLR 2021) and
/// lmnt-com/wavegrad for what the paper leaves unstated.</para>
/// <para>
/// Training (Algorithm 1, §2.2): a segment s uniform over 1..S, a level √ᾱ uniform between l_{s−1} and l_s
/// (<c>l_0 = 1</c>, <c>l_s = √Π_{i≤s}(1 − β_i)</c>), <c>y = √ᾱ y₀ + √(1 − ᾱ) ε</c> and the L1 loss
/// <c>‖ε − ε_θ(y, x, √ᾱ)‖₁</c>. Synthesis (Algorithm 2) runs the reverse process over any β schedule, conditioning each
/// step on that schedule's √ᾱ_n, so one model serves the 1000-step and the six-step schedules alike.
/// </para>
/// <para><b>For Beginners:</b> WaveGrad starts from noise and repeatedly nudges it toward speech, each step following
/// the network's estimate of which direction makes the audio more like real speech for this spectrogram.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 128, outputSize: 300);
///
/// // The paper's WaveGrad Base, sampled with a short six-step schedule.
/// var model = new WaveGrad&lt;double&gt;(architecture,
///     new WaveGradOptions { InferenceNoiseSchedule = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1] });
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "WaveGrad: Estimating Gradients for Waveform Generation",
    "https://arxiv.org/abs/2009.00713",
    Year = 2021,
    Authors = "Chen et al."
)]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 2e-4, ReferenceBatchSize = 256,
                Source = "Chen et al. 2021, Sec. 4: batch 256, about 1M steps; the paper states no optimizer, so Adam at 2e-4 with the gradient norm clipped to 1 follows lmnt-com/wavegrad params.py.")]
public partial class WaveGrad<T> : DiffusionVocoderBase<T>
{
    private WaveGradNetwork<T>? _network;
    private CenteredLogMel<T>? _features;
    private double[]? _levels;

    /// <summary>Creates a WaveGrad that runs an exported ONNX graph.</summary>
    public WaveGrad(NeuralNetworkArchitecture<T> architecture, string modelPath, WaveGradOptions? options = null)
        : base(architecture, modelPath, options ?? new WaveGradOptions(), options?.SamplingSeed ?? 0)
    {
    }

    /// <summary>Creates a trainable WaveGrad.</summary>
    public WaveGrad(NeuralNetworkArchitecture<T> architecture, WaveGradOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new WaveGradOptions(), optimizer, options?.SamplingSeed ?? 0)
    {
    }

    private WaveGradOptions PaperOptions => (WaveGradOptions)VocoderSettings;

    /// <inheritdoc />
    public override int UpsampleFactor => PaperOptions.UpsampleFactors.Aggregate(1, (a, b) => a * b);

    /// <inheritdoc />
    protected override int SegmentSize => PaperOptions.CropFrames * UpsampleFactor;

    /// <inheritdoc />
    protected override double[] TrainingBetas => PaperOptions.NoiseSchedule;

    /// <inheritdoc />
    protected override double[]? InferenceBetas => PaperOptions.InferenceNoiseSchedule;

    /// <inheritdoc />
    protected override bool ClampEachStep => true;

    /// <inheritdoc />
    protected override double GradientClipNorm => PaperOptions.GradientClipNorm;

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateNetwork()
    {
        var o = PaperOptions;
        if (UpsampleFactor != o.HopSize)
            throw new ArgumentException($"The UBlocks' factors ({string.Join("x", o.UpsampleFactors)}) must multiply to the hop ({o.HopSize}).");
        _network = new WaveGradNetwork<T>(Engine, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed + 7),
            o.MelChannels, o.MelProjectionChannels, o.WaveformChannels, o.UpsampleFactors, o.UpsampleChannels, o.UpsampleDilations,
            o.RepeatBlocks, o.LeakySlope, o.NoiseLevelScale);
        _features = new CenteredLogMel<T>(Engine, o.SampleRate, o.FftSize, o.HopSize, o.WindowSize, o.MelChannels, o.MelMinFrequency,
            o.MelMaxFrequency, 1e-5, htkScale: true, normalizedWindow: true);
        return _network.Layers;
    }

    /// <inheritdoc />
    protected override Tensor<T> Denoise(Tensor<T> noisy, double level, Tensor<T> mel) => _network!.Forward(noisy, level, mel);

    /// <inheritdoc />
    /// <remarks>Hierarchical sampling (§2.2, Eq. 12): a segment s uniform over 1..S, then √ᾱ uniform over
    /// (l_{s−1}, l_s); the network is conditioned on that √ᾱ itself.</remarks>
    protected override (double Level, double SqrtAlphaBar) DrawTrainingLevel(Random random)
    {
        if (_levels is null)
        {
            var cumulative = Cumulative(TrainingBetas);
            _levels = new double[cumulative.Length + 1];
            _levels[0] = 1;
            for (int s = 0; s < cumulative.Length; s++) _levels[s + 1] = Math.Sqrt(cumulative[s]);
        }
        int segment = random.Next(1, _levels.Length);
        double level = _levels[segment - 1] + random.NextDouble() * (_levels[segment] - _levels[segment - 1]);
        return (level, level);
    }

    /// <inheritdoc />
    /// <remarks>Each step is conditioned on its own schedule's noise level √ᾱ_n (Algorithm 2).</remarks>
    protected override double InferenceLevel(int n, double alphaBar) => Math.Sqrt(alphaBar);

    /// <inheritdoc />
    /// <remarks>The L1 distance (§2.1: "substituting the original L2 distance metric with L1 offers better training
    /// stability").</remarks>
    protected override Tensor<T> NoiseLoss(Tensor<T> noise, Tensor<T> predicted, Tensor<T> mel)
        => Mean(Engine.TensorAbs(Engine.TensorSubtract(noise, predicted)));

    /// <inheritdoc />
    protected override Tensor<T> ComputeInputMel(Tensor<T> audio) => DecibelUnitRange(_features!.Forward(audio));

    /// <inheritdoc />
    protected override void OnMutableConstructorConfigurationRestored()
    {
        base.OnMutableConstructorConfigurationRestored();
        _levels = null;
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
            Name = IsOnnxMode ? "WaveGrad-ONNX" : "WaveGrad-Native",
            Description = "WaveGrad: Estimating Gradients for Waveform Generation (Chen et al., 2021)",
            FeatureCount = o.MelChannels,
            Complexity = o.UpsampleFactors.Length,
        };
        m.AdditionalInfo["Architecture"] = o.RepeatBlocks ? "WaveGrad-Large" : "WaveGrad-Base";
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        return m;
    }
}
