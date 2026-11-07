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
/// FreGrad: a lightweight and fast frequency-aware diffusion vocoder that denoises the waveform's two Haar wavelet
/// sub-bands with frequency-aware dilated convolutions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "FreGrad: Lightweight and Fast Frequency-aware Diffusion Vocoder" (Nguyen et al., ICASSP
/// 2024) and kaistmm/fregrad for what the paper leaves unstated.</para>
/// <para>
/// The waveform is split by the Haar DWT into low and high sub-bands of half its length (§3.1); each band is diffused
/// with noise from its own PriorGrad-style energy prior, from the lower or the upper half of the mel bands (§3.3); a
/// DiffWave network whose dilated convolutions are Freq-DConvs (§3.2), taking and predicting both bands, estimates the
/// noise; the loss is Σ_{l,h} (‖ε − ε̂‖²_{Σ⁻¹} + λ L_mag(ε, ε̂)) with L_mag the multi-resolution STFT log-magnitude
/// loss (Eq. 9–10); the β schedule is shifted to zero terminal SNR (Eq. 8). Sampling denoises both bands from their
/// priors and returns their inverse DWT.
/// </para>
/// <para><b>For Beginners:</b> FreGrad splits audio into a low-pitched and a high-pitched half-length signal, which are
/// simpler to clean up than the full waveform, removes noise from both, and recombines them losslessly.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "FreGrad: Lightweight and Fast Frequency-aware Diffusion Vocoder",
    "https://arxiv.org/abs/2401.10032",
    Year = 2024,
    Authors = "Nguyen et al."
)]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 2e-4, Beta1 = 0.9, Beta2 = 0.999, ReferenceBatchSize = 16,
                Source = "Nguyen et al. 2024, Sec. 4.1: Adam with beta1 = 0.9, beta2 = 0.999, a fixed learning rate of 0.0002 and a batch size of 16.")]
public partial class FreGrad<T> : DiffusionVocoderBase<T>
{
    private DiffWaveNetwork<T>? _network;
    private DifferentiableMel<T>? _features;
    private MultiResolutionStftLoss<T>? _stftLoss;

    /// <summary>Creates a FreGrad that runs an exported ONNX graph.</summary>
    public FreGrad(NeuralNetworkArchitecture<T> architecture, string modelPath, FreGradOptions? options = null)
        : base(architecture, modelPath, options ?? new FreGradOptions(), options?.SamplingSeed ?? 0)
    {
    }

    /// <summary>Creates a trainable FreGrad.</summary>
    public FreGrad(NeuralNetworkArchitecture<T> architecture, FreGradOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new FreGradOptions(), optimizer, options?.SamplingSeed ?? 0)
    {
    }

    private FreGradOptions PaperOptions => (FreGradOptions)VocoderSettings;

    /// <inheritdoc />
    /// <remarks>Twice the mel upsampler's factor: the network runs on half-length sub-bands.</remarks>
    public override int UpsampleFactor => 2 * PaperOptions.UpsampleStrides.Aggregate(1, (a, b) => a * b);

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
            throw new ArgumentException($"The upsampler's strides ({string.Join("x", o.UpsampleStrides)}) must multiply to half the hop ({o.HopSize}).");
        if (o.MelChannels < 2)
            throw new ArgumentException("The two priors need at least two mel bands.");
        _network = new DiffWaveNetwork<T>(Engine, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed + 7),
            o.MelChannels, o.ResChannels, o.NumResLayers, o.DilationCycle, o.NoiseSchedule.Length, o.UpsampleStrides,
            audioChannels: 2, frequencyAware: true);
        _features = new DifferentiableMel<T>(Engine, o.SampleRate, o.FftSize, o.HopSize, o.WindowSize, o.MelChannels,
            o.MelMinFrequency, o.MelMaxFrequency);
        _stftLoss = new MultiResolutionStftLoss<T>(Engine, o.StftFftSizes, o.StftHopSizes, o.StftWindowSizes);
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

    /// <inheritdoc />
    /// <remarks>The low and high Haar sub-bands as two channels <c>[1, 2, samples / 2]</c>.</remarks>
    protected override Tensor<T> ToDiffusionSpace(Tensor<T> waveform)
    {
        var (low, high) = HaarWavelet.Forward(Engine, waveform);
        return Engine.TensorConcatenate(new[] { low, high }, 1);
    }

    /// <inheritdoc />
    protected override Tensor<T> FromDiffusionSpace(Tensor<T> sample)
    {
        int n = sample.Shape[2];
        return HaarWavelet.Inverse(Engine, Engine.TensorSlice(sample, new[] { 0, 0, 0 }, new[] { 1, 1, n }),
            Engine.TensorSlice(sample, new[] { 0, 1, 0 }, new[] { 1, 1, n }));
    }

    /// <summary>The priors' standard deviations <c>[1, 2, frames · hop / 2]</c> for a mel spectrogram
    /// <c>[1, mel, frames]</c>: the low band's from the lower half of the mel bands, the high band's from the upper half,
    /// each normalized as PriorGrad's and repeated over the frame's half-hop.</summary>
    public Tensor<T> PriorStd(Tensor<T> mel)
    {
        var o = PaperOptions;
        mel = MelInput(mel);
        int half = o.MelChannels / 2, repeat = UpsampleFactor / 2;
        var low = EnergyPrior.Std<T>(EnergyPrior.FrameEnergies(mel, 0, half), o, repeat);
        var high = EnergyPrior.Std<T>(EnergyPrior.FrameEnergies(mel, half, o.MelChannels), o, repeat);
        return Engine.TensorConcatenate(new[] { low, high }, 1);
    }

    /// <summary>Sets the energy statistics of the priors from training recordings (see
    /// <see cref="PriorGrad{T}.FitEnergyStatistics"/>).</summary>
    public void FitEnergyStatistics(IEnumerable<Tensor<T>> recordings, bool useDataMaximum = false)
    {
        if (recordings is null) throw new ArgumentNullException(nameof(recordings));
        EnergyPrior.Fit(recordings.Select(ComputeMel), PaperOptions, useDataMaximum);
    }

    /// <inheritdoc />
    /// <remarks>ε ~ N(0, diag(σ_l², σ_h²)) over the two sub-bands.</remarks>
    protected override Tensor<T> PriorNoise(Tensor<T> mel, int samples, Random random)
        => Engine.TensorMultiply(Gaussian(new[] { 1, 2, samples / 2 }, random), PriorStd(mel));

    /// <inheritdoc />
    /// <remarks>Σ over the two bands of the Mahalanobis noise loss plus λ times the STFT log-magnitude loss between
    /// the band's true and predicted noise (Eq. 10).</remarks>
    protected override Tensor<T> NoiseLoss(Tensor<T> noise, Tensor<T> predicted, Tensor<T> mel)
    {
        Tensor<T> inverse;
        using (new NoGradScope<T>())
            inverse = Engine.TensorReciprocal(PriorStd(mel));
        int n = noise.Shape[2];
        Tensor<T>? total = null;
        for (int band = 0; band < 2; band++)
        {
            var start = new[] { 0, band, 0 };
            var length = new[] { 1, 1, n };
            var e = Engine.TensorSlice(noise, start, length);
            var p = Engine.TensorSlice(predicted, start, length);
            var d = Engine.TensorMultiply(Engine.TensorSubtract(e, p), Engine.TensorSlice(inverse, start, length));
            var diffusion = Mean(Engine.TensorMultiply(d, d));
            var magnitude = _stftLoss!.Forward(new[] { Engine.Reshape(p, new[] { n }) }, new[] { Engine.Reshape(e, new[] { n }) }).LogMagnitude;
            var term = Engine.TensorAdd(diffusion, Engine.TensorMultiplyScalar(magnitude, NumOps.FromDouble(PaperOptions.MagnitudeLossWeight)));
            total = total is null ? term : Engine.TensorAdd(total, term);
        }
        return total!;
    }

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer()
        => PaperOptimizerFactory.VerifyHandBuilt(this, new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = PaperOptions.LearningRate,
                Beta1 = 0.9,
                Beta2 = 0.999,
                UseAdaptiveBetas = false,
            }));

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = PaperOptions;
        var m = new ModelMetadata<T>
        {
            Name = IsOnnxMode ? "FreGrad-ONNX" : "FreGrad-Native",
            Description = "FreGrad: Lightweight and Fast Frequency-aware Diffusion Vocoder (Nguyen et al., 2024)",
            FeatureCount = o.MelChannels,
            Complexity = o.NumResLayers,
        };
        m.AdditionalInfo["Architecture"] = "FreGrad";
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        return m;
    }
}
