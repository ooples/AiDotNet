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
/// PriorGrad vocoder: DiffWave whose diffusion prior is a data-dependent diagonal Gaussian N(0, Σ_c) with the
/// standard deviation of each frame taken from the mel spectrogram's normalized frame energy.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "PriorGrad: Improving Conditional Denoising Diffusion Models with Data-Dependent Adaptive
/// Prior" (Lee et al., ICLR 2022) and microsoft/NeuralSpeech PriorGrad-vocoder for what the paper leaves unstated.</para>
/// <para>
/// Training (Algorithm 1): ε ~ N(0, Σ), t uniform, <c>x_t = √ᾱ_t x₀ + √(1 − ᾱ_t) ε</c> and the Mahalanobis loss
/// <c>‖ε − ε_θ(x_t, c, t)‖²_{Σ⁻¹}</c>. Sampling (Algorithm 2): x_T ~ N(0, Σ) and every step's noise z ~ N(0, Σ). Σ is
/// the frame energy √Σ_bands exp(mel) normalized to (0, 1], clipped below at 0.1 and repeated over each frame's hop
/// (§4). The network is DiffWave's, unchanged.
/// </para>
/// <para><b>For Beginners:</b> DiffWave starts every synthesis from the same plain noise; PriorGrad starts from noise
/// that is already loud where the speech is loud and quiet where it is quiet, so there is less left to learn.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "PriorGrad: Improving Conditional Denoising Diffusion Models with Data-Dependent Adaptive Prior",
    "https://arxiv.org/abs/2106.06406",
    Year = 2022,
    Authors = "Lee et al."
)]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 2e-4, ReferenceBatchSize = 16,
                Source = "Lee et al. 2022, Sec. 4: Adam at 2e-4 for 1M iterations; batch 16 from the PriorGrad-vocoder params.py.")]
public partial class PriorGrad<T> : DiffusionVocoderBase<T>
{
    private DiffWaveNetwork<T>? _network;
    private DifferentiableMel<T>? _features;

    /// <summary>Creates a PriorGrad that runs an exported ONNX graph.</summary>
    public PriorGrad(NeuralNetworkArchitecture<T> architecture, string modelPath, PriorGradOptions? options = null)
        : base(architecture, modelPath, options ?? new PriorGradOptions(), options?.SamplingSeed ?? 0)
    {
    }

    /// <summary>Creates a trainable PriorGrad.</summary>
    public PriorGrad(NeuralNetworkArchitecture<T> architecture, PriorGradOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new PriorGradOptions(), optimizer, options?.SamplingSeed ?? 0)
    {
    }

    private PriorGradOptions PaperOptions => (PriorGradOptions)VocoderSettings;

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
        _features = new DifferentiableMel<T>(Engine, o.SampleRate, o.FftSize, o.HopSize, o.WindowSize, o.MelChannels,
            o.MelMinFrequency, o.MelMaxFrequency);
        return _network.Layers;
    }

    /// <inheritdoc />
    protected override Tensor<T> Denoise(Tensor<T> noisy, double level, Tensor<T> mel) => _network!.Forward(noisy, level, mel);

    /// <inheritdoc />
    /// <remarks>The natural-log mel spectrogram <c>ln(max(mel, 1e-5))</c> of the HiFi-GAN pipeline.</remarks>
    protected override Tensor<T> ComputeInputMel(Tensor<T> audio)
    {
        var rows = _features!.Forward(audio);
        return Engine.Reshape(Engine.TensorTranspose(rows), new[] { 1, PaperOptions.MelChannels, rows.Shape[0] });
    }

    /// <summary>The prior's standard deviation per sample <c>[1, 1, samples]</c> for a mel spectrogram
    /// <c>[1, mel, frames]</c>: each frame's energy normalized to (0, 1], clipped below at the minimum standard deviation,
    /// repeated over the hop.</summary>
    public Tensor<T> PriorStd(Tensor<T> mel)
    {
        mel = MelInput(mel);
        return EnergyPrior.Std<T>(EnergyPrior.FrameEnergies(mel, 0, PaperOptions.MelChannels), PaperOptions, UpsampleFactor);
    }

    /// <summary>Sets <see cref="PriorGradOptions.EnergyMin"/> (and, with <paramref name="useDataMaximum"/>,
    /// <see cref="PriorGradOptions.EnergyMax"/>) to the extremes of the recordings' frame energies, as the reference
    /// computes them over the training set.</summary>
    /// <param name="recordings">The training waveforms.</param>
    /// <param name="useDataMaximum">Whether the maximum also comes from the data rather than the override of 4.</param>
    public void FitEnergyStatistics(IEnumerable<Tensor<T>> recordings, bool useDataMaximum = false)
    {
        if (recordings is null) throw new ArgumentNullException(nameof(recordings));
        EnergyPrior.Fit(recordings.Select(ComputeMel), PaperOptions, useDataMaximum);
    }

    /// <inheritdoc />
    /// <remarks>ε ~ N(0, Σ_c): standard-normal noise scaled by the prior's standard deviation.</remarks>
    protected override Tensor<T> PriorNoise(Tensor<T> mel, int samples, Random random)
        => Engine.TensorMultiply(Gaussian(new[] { 1, 1, samples }, random), PriorStd(mel));

    /// <inheritdoc />
    /// <remarks>The Mahalanobis distance under the diagonal prior, <c>mean(((ε − ε_θ) / σ)²)</c>.</remarks>
    protected override Tensor<T> NoiseLoss(Tensor<T> noise, Tensor<T> predicted, Tensor<T> mel)
    {
        Tensor<T> inverse;
        using (new NoGradScope<T>())
            inverse = Engine.TensorReciprocal(PriorStd(mel));
        var d = Engine.TensorMultiply(Engine.TensorSubtract(noise, predicted), inverse);
        return Mean(Engine.TensorMultiply(d, d));
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
            Name = IsOnnxMode ? "PriorGrad-ONNX" : "PriorGrad-Native",
            Description = "PriorGrad: Improving Conditional Denoising Diffusion Models with Data-Dependent Adaptive Prior (Lee et al., 2022)",
            FeatureCount = o.MelChannels,
            Complexity = o.NumResLayers,
        };
        m.AdditionalInfo["Architecture"] = "PriorGrad";
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        return m;
    }
}

/// <summary>
/// PriorGrad's data-dependent prior (Lee et al. 2022, §4; reference <c>dataset.py</c>), shared with FreGrad: the frame
/// energy √Σ exp(mel) over a range of mel bands, normalized by the training-set extremes to (0, 1], clipped below at the
/// minimum standard deviation and repeated over the frame's samples.
/// </summary>
internal static class EnergyPrior
{
    /// <summary>The frame energies √Σ_{bands from..to} exp(mel) of a mel spectrogram <c>[1, mel, frames]</c>.</summary>
    public static double[] FrameEnergies<T>(Tensor<T> mel, int fromBand, int toBand)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int frames = mel.Shape[2];
        var energy = new double[frames];
        for (int f = 0; f < frames; f++)
        {
            double sum = 0;
            for (int m = fromBand; m < toBand; m++) sum += Math.Exp(ops.ToDouble(mel[0, m, f]));
            energy[f] = Math.Sqrt(sum);
        }
        return energy;
    }

    /// <summary>The standard deviation <c>[1, 1, frames · repeat]</c> of the frame energies: (min(e, max) − min) /
    /// (max − min), at least the minimum standard deviation, each repeated <paramref name="repeat"/> times. Without a
    /// fitted minimum it is the energy of silence over every band, √(bands · 1e-5).</summary>
    public static Tensor<T> Std<T>(double[] energy, PriorGradOptions o, int repeat)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        double min = o.EnergyMin ?? Math.Sqrt(o.MelChannels * 1e-5), max = o.EnergyMax;
        if (!(max > min))
            throw new InvalidOperationException($"The energy range is empty (min {min}, max {max}).");
        var std = new Tensor<T>(new[] { 1, 1, energy.Length * repeat });
        for (int f = 0; f < energy.Length; f++)
        {
            double s = Math.Max((Math.Min(energy[f], max) - min) / (max - min), o.MinStd);
            for (int i = 0; i < repeat; i++) std[0, 0, f * repeat + i] = ops.FromDouble(s);
        }
        return std;
    }

    /// <summary>Sets the options' energy minimum (and, with <paramref name="useDataMaximum"/>, maximum) to the extremes
    /// of the full-band frame energies of <paramref name="mels"/>.</summary>
    public static void Fit<T>(IEnumerable<Tensor<T>> mels, PriorGradOptions o, bool useDataMaximum)
    {
        double min = double.PositiveInfinity, max = double.NegativeInfinity;
        foreach (var mel in mels)
            foreach (var e in FrameEnergies(mel, 0, mel.Shape[1]))
            {
                min = Math.Min(min, e);
                max = Math.Max(max, e);
            }
        if (double.IsInfinity(min))
            throw new ArgumentException("No frames to measure.", nameof(mels));
        o.EnergyMin = min;
        if (useDataMaximum) o.EnergyMax = max;
    }
}
