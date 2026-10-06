using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// The frame-level amplitude-and-phase vocoders' shared generator and losses (APNet, Ai and Ling 2023; APNet2, Du et al.
/// 2023): an amplitude spectrum predictor (ASP) and a phase spectrum predictor (PSP) whose two parallel outputs give the
/// phase by the two-argument arctangent, an inverse STFT, and the multilevel loss λ_A L_A + λ_P L_P + λ_S L_S + L_W.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <c>log Â = ASP(M)</c>, <c>P̂ = Φ(R, I)</c>, <c>x̂ = ISTFT(Â e^{jP̂})</c>. L_A is the log-amplitude MSE; L_P the
/// instantaneous-phase, group-delay and phase-time-difference errors through the model's anti-wrapping function; L_S the
/// STFT consistency loss between Ŝ and STFT(x̂) plus λ_RI times the L1 real and imaginary part losses; L_W the model's
/// adversarial and feature-matching terms plus λ_Mel times the L1 mel loss.
/// </para>
/// <para><b>For Beginners:</b> These vocoders predict how loud each frequency is and where each wave is in its cycle for
/// every frame, then an inverse Fourier transform turns that into audio.</para>
/// </remarks>
public abstract partial class AmplitudePhaseVocoderBase<T> : GanVocoderBase<T>
{
    private CenteredLogMel<T>? _features;
    private CenteredComplexStft<T>? _stft;
    private InverseStft<T>? _istft;

    /// <summary>Creates a native (trainable) vocoder.</summary>
    protected AmplitudePhaseVocoderBase(NeuralNetworkArchitecture<T> architecture, AmplitudePhaseVocoderOptions options,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer, int samplingSeed)
        : base(architecture, options, optimizer, samplingSeed)
    {
    }

    /// <summary>Creates a vocoder that runs an exported ONNX graph.</summary>
    protected AmplitudePhaseVocoderBase(NeuralNetworkArchitecture<T> architecture, string modelPath, AmplitudePhaseVocoderOptions options,
        int samplingSeed)
        : base(architecture, modelPath, options, samplingSeed)
    {
    }

    private AmplitudePhaseVocoderOptions Settings => (AmplitudePhaseVocoderOptions)VocoderSettings;

    // ---------------------------------------------------------------- hooks

    /// <summary>Builds the ASP and PSP into <paramref name="layers"/>.</summary>
    protected abstract void CreatePredictors(List<LayerBase<T>> layers, Random initialization);

    /// <summary>The ASP's log amplitude and the PSP's pseudo real and imaginary parts <c>[1, bins, frames]</c> of a mel
    /// spectrogram <c>[1, mel, frames]</c>.</summary>
    protected abstract (Tensor<T> LogAmplitude, Tensor<T> Real, Tensor<T> Imaginary) PredictComponents(Tensor<T> mel);

    /// <summary>The anti-wrapping error of each element of a phase difference (APNet: −cos; APNet2: |x − 2π round(x/2π)|).</summary>
    protected abstract Tensor<T> PhaseError(Tensor<T> difference);

    /// <summary>The adversarial and feature-matching terms of the generator loss for the generated waveform.</summary>
    protected abstract Tensor<T> AdversarialTerms(Tensor<T> generated, Tensor<T> real);

    // ---------------------------------------------------------------- generator

    /// <inheritdoc />
    public override int UpsampleFactor => Settings.HopSize;

    /// <inheritdoc />
    /// <remarks>The reference synthesizes with <c>torch.istft(center=True)</c>: T frames give <c>(T − 1) · hop</c> samples.</remarks>
    protected override int WaveformLengthOffset => -UpsampleFactor;

    /// <inheritdoc />
    protected override int SegmentSize => Settings.SegmentSize;

    /// <inheritdoc />
    /// <remarks>F centred frames give (F − 1) · hop samples through the centred inverse STFT.</remarks>
    protected override bool KeepsCentredFrame => true;

    /// <inheritdoc />
    protected override IReadOnlyList<LayerBase<T>> CreateGenerator()
    {
        var o = Settings;
        var layers = new List<LayerBase<T>>();
        CreatePredictors(layers, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed + 7));
        _features = new CenteredLogMel<T>(Engine, o.SampleRate, o.FftSize, o.HopSize, o.WindowSize, o.MelChannels, 0.0, o.MelMaxFrequency, 1e-5,
            naturalLog: true);
        _stft = new CenteredComplexStft<T>(Engine, o.FftSize, o.HopSize, o.WindowSize);
        _istft = new InverseStft<T>(Engine, o.FftSize, o.HopSize, o.WindowSize);
        return layers;
    }

    /// <inheritdoc />
    /// <remarks>The references' <c>mel_spectrogram</c>: a centred STFT, Slaney bands up to the mel maximum,
    /// <c>ln(max(x, 1e-5))</c>.</remarks>
    protected override Tensor<T> ComputeInputMel(Tensor<T> audio) => _features!.Forward(audio);

    /// <summary>The log amplitude and phase spectra <c>[1, bins, frames]</c> the ASP and PSP predict for a mel spectrogram
    /// <c>[1, mel, frames]</c>; the phase is Φ(R, I), the two-argument arctangent (Eq. 5).</summary>
    public (Tensor<T> LogAmplitude, Tensor<T> Phase) PredictSpectra(Tensor<T> mel)
    {
        var (logAmplitude, real, imaginary) = PredictComponents(MelInput(mel));
        return (logAmplitude, Engine.TensorAtan2(imaginary, real));
    }

    /// <summary>The natural log amplitude <c>ln(|S| + 1e-5)</c>, phase, real and imaginary parts <c>[1, bins, frames]</c>
    /// of a waveform <c>[samples]</c> (reference <c>amp_pha_specturm</c>).</summary>
    public (Tensor<T> LogAmplitude, Tensor<T> Phase, Tensor<T> Re, Tensor<T> Im) AnalyzeSpectra(Tensor<T> audio)
    {
        using var _ = new NoGradScope<T>();
        var (re, im) = _stft!.Forward(Flat(audio));
        re = Detached(re);
        im = Detached(im);
        var magnitude = Engine.TensorPow(Engine.TensorAdd(Engine.TensorMultiply(re, re), Engine.TensorMultiply(im, im)), NumOps.FromDouble(0.5));
        return (Detached(Engine.TensorLog(Engine.TensorAddScalar(magnitude, NumOps.FromDouble(1e-5)))), Detached(Engine.TensorAtan2(im, re)), re, im);
    }

    /// <inheritdoc />
    protected override Tensor<T> Generate(Tensor<T> mel)
    {
        var (logAmplitude, phase) = PredictSpectra(mel);
        var wave = _istft!.Forward(Engine.TensorExp(logAmplitude), phase);
        return Engine.Reshape(wave, new[] { 1, 1, wave.Length });
    }

    // ---------------------------------------------------------------- losses

    private Tensor<T> PhaseTerm(Tensor<T> real, Tensor<T> predicted) => Mean(PhaseError(Engine.TensorSubtract(real, predicted)));

    /// <summary>λ_A L_A + λ_P L_P + λ_S L_S and the generated waveform, for a mel spectrogram and the waveform it was
    /// analysed from.</summary>
    private (Tensor<T> Loss, Tensor<T> Generated) FrameLevelLosses(Tensor<T> mel, Tensor<T> real)
    {
        var o = Settings;
        var (logAmplitude, phase) = PredictSpectra(mel);
        var amplitude = Engine.TensorExp(logAmplitude);
        var re = Engine.TensorMultiply(amplitude, Engine.TensorCos(phase));
        var im = Engine.TensorMultiply(amplitude, Engine.TensorSin(phase));
        var generated = _istft!.Forward(amplitude, phase);
        var (realLog, realPhase, realRe, realIm) = AnalyzeSpectra(real);

        var d = Engine.TensorSubtract(logAmplitude, realLog);
        var amplitudeLoss = Mean(Engine.TensorMultiply(d, d));
        var phaseLoss = Sum(new[]
        {
            PhaseTerm(realPhase, phase),
            PhaseTerm(PhaseDifferences.Along(Engine, realPhase, 1), PhaseDifferences.Along(Engine, phase, 1)),
            PhaseTerm(PhaseDifferences.Along(Engine, realPhase, 2), PhaseDifferences.Along(Engine, phase, 2)),
        });
        var (consistentRe, consistentIm) = _stft!.Forward(generated);
        var gapRe = Engine.TensorSubtract(re, consistentRe);
        var gapIm = Engine.TensorSubtract(im, consistentIm);
        var consistency = Mean(Engine.TensorAdd(Engine.TensorMultiply(gapRe, gapRe), Engine.TensorMultiply(gapIm, gapIm)));
        var realImaginary = Engine.TensorAdd(Mean(Engine.TensorAbs(Engine.TensorSubtract(realRe, re))),
            Mean(Engine.TensorAbs(Engine.TensorSubtract(realIm, im))));
        var stftLoss = Engine.TensorAdd(consistency, Engine.TensorMultiplyScalar(realImaginary, NumOps.FromDouble(o.RealImaginaryLossWeight)));
        var loss = Sum(new[]
        {
            Engine.TensorMultiplyScalar(amplitudeLoss, NumOps.FromDouble(o.AmplitudeLossWeight)),
            Engine.TensorMultiplyScalar(phaseLoss, NumOps.FromDouble(o.PhaseLossWeight)),
            Engine.TensorMultiplyScalar(stftLoss, NumOps.FromDouble(o.StftLossWeight)),
        });
        return (loss, generated);
    }

    private Tensor<T> MelLoss(Tensor<T> mel, Tensor<T> generated)
    {
        Tensor<T> target;
        using (new NoGradScope<T>()) target = Detached(mel);
        return Engine.TensorMultiplyScalar(Mean(Engine.TensorAbs(Engine.TensorSubtract(_features!.Forward(generated), target))),
            NumOps.FromDouble(Settings.MelLossWeight));
    }

    /// <inheritdoc />
    /// <remarks>λ_A L_A + λ_P L_P + λ_S L_S + λ_Mel L_Mel, plus the adversarial and feature-matching terms once the
    /// discriminators train.</remarks>
    protected override Tensor<T> GeneratorLoss(Tensor<T> mel, Tensor<T> real, bool adversarial)
    {
        var target = Flat(real);
        var (frameLevel, generated) = FrameLevelLosses(mel, target);
        var loss = Engine.TensorAdd(frameLevel, MelLoss(mel, generated));
        return adversarial ? Engine.TensorAdd(loss, AdversarialTerms(generated, target)) : loss;
    }

    /// <inheritdoc />
    /// <remarks>The generator loss without its adversarial and feature-matching terms.</remarks>
    protected override Tensor<T> ReconstructionObjective(Tensor<T> mel, Tensor<T> real)
    {
        var (frameLevel, generated) = FrameLevelLosses(mel, Flat(real));
        return Engine.TensorAdd(frameLevel, MelLoss(mel, generated));
    }

    /// <inheritdoc />
    /// <remarks>AdamW (β = 0.8, 0.99, weight decay 0.01) at 2e-4 for each network, decayed by the per-epoch factor once
    /// per epoch of updates.</remarks>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer(string group)
    {
        var o = Settings;
        int perEpoch = Math.Max(1, o.UpdatesPerEpoch);
        var scheduler = new AiDotNet.LearningRateSchedulers.LambdaLRScheduler(o.LearningRate,
            step => Math.Pow(o.LearningRateDecay, step / perEpoch));
        return PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = o.LearningRate,
                Beta1 = 0.8,
                Beta2 = 0.99,
                WeightDecay = o.WeightDecay,
                LearningRateScheduler = scheduler,
                SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
            }));
    }
}
