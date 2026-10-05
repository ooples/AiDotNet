using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>
/// The denoising-diffusion training and sampling diffusion vocoders share (Ho et al. 2020; DiffWave, WaveGrad, PriorGrad,
/// FreGrad): a network ε_θ predicts the noise added to a waveform at a noise level, conditioned on the mel spectrogram,
/// and the reverse process turns noise into speech.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Training (Algorithm 1 of DDPM and the vocoder papers): crop a segment, draw a noise level, form
/// <c>x_t = √ᾱ_t x₀ + √(1 − ᾱ_t) ε</c> and minimize the model's loss between ε and ε_θ(x_t, level, mel). Sampling
/// (Algorithm 2): from x_N ~ the prior, <c>x_{n−1} = (x_n − β_n / √(1 − ᾱ_n) · ε_θ) / √α_n + σ_n z</c> with
/// <c>σ_n = √((1 − ᾱ_{n−1}) / (1 − ᾱ_n) · β_n)</c>, over the training schedule or a shorter inference schedule.
/// </para>
/// <para>A model supplies its network (<see cref="Denoise"/>), its input features, its schedules and how the noise level
/// conditions the network, and may change the noise prior and the loss (PriorGrad's data-dependent prior and Mahalanobis
/// loss) and the space the diffusion runs in (FreGrad's wavelet sub-bands).</para>
/// <para><b>For Beginners:</b> The model learns to remove a little noise at a time; to make speech it starts from pure
/// noise and removes noise step by step, guided by the spectrogram.</para>
/// </remarks>
public abstract partial class DiffusionVocoderBase<T> : SegmentVocoderBase<T>
{
    /// <summary>Creates a native (trainable) vocoder.</summary>
    protected DiffusionVocoderBase(NeuralNetworkArchitecture<T> architecture, VocoderOptions options,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer, int samplingSeed)
        : base(architecture, options, optimizer, samplingSeed)
    {
    }

    /// <summary>Creates a vocoder that runs an exported ONNX graph.</summary>
    protected DiffusionVocoderBase(NeuralNetworkArchitecture<T> architecture, string modelPath, VocoderOptions options, int samplingSeed)
        : base(architecture, modelPath, options, samplingSeed)
    {
    }

    // ---------------------------------------------------------------- hooks

    /// <summary>ε_θ(x, level, mel): the predicted noise for the noisy diffusion-space sample <paramref name="noisy"/>
    /// (<c>[1, 1, samples]</c> unless <see cref="ToDiffusionSpace"/> says otherwise) at the conditioning value
    /// <paramref name="level"/> (a diffusion step for DiffWave, a noise level for WaveGrad) given the mel spectrogram
    /// <c>[1, mel, frames]</c>.</summary>
    protected abstract Tensor<T> Denoise(Tensor<T> noisy, double level, Tensor<T> mel);

    /// <summary>The training noise schedule β₁..β_T.</summary>
    protected abstract double[] TrainingBetas { get; }

    /// <summary>The inference noise schedule, or null to sample with the training schedule.</summary>
    protected virtual double[]? InferenceBetas => null;

    /// <summary>The conditioning value and √ᾱ of a training draw. The default draws a step t uniformly and conditions
    /// on its index (DiffWave).</summary>
    protected virtual (double Level, double SqrtAlphaBar) DrawTrainingLevel(Random random)
    {
        var cumulative = Cumulative(TrainingBetas);
        int t = random.Next(cumulative.Length);
        return (t, Math.Sqrt(cumulative[t]));
    }

    /// <summary>The conditioning value of inference step <paramref name="n"/> whose ᾱ is <paramref name="alphaBar"/>.
    /// The default aligns ᾱ with the training schedule and interpolates the step index between the two training steps
    /// around it (DiffWave App. B; reference <c>inference.py</c>).</summary>
    protected virtual double InferenceLevel(int n, double alphaBar)
    {
        var training = Cumulative(TrainingBetas);
        if (InferenceBetas is null) return n;
        double s = Math.Sqrt(alphaBar);
        for (int t = 0; t < training.Length - 1; t++)
            if (training[t + 1] <= alphaBar && alphaBar <= training[t])
                return t + (Math.Sqrt(training[t]) - s) / (Math.Sqrt(training[t]) - Math.Sqrt(training[t + 1]));
        return alphaBar >= training[0] ? 0 : training.Length - 1;
    }

    /// <summary>Noise of the prior, in the diffusion space, for a waveform of <paramref name="samples"/> samples given its
    /// mel spectrogram; the default is the standard normal over <c>[1, 1, samples]</c>.</summary>
    protected virtual Tensor<T> PriorNoise(Tensor<T> mel, int samples, Random random) => Gaussian(new[] { 1, 1, samples }, random);

    /// <summary>The diffusion space's representation of a waveform <c>[1, 1, samples]</c>; the default is the waveform
    /// itself (FreGrad diffuses its Haar wavelet sub-bands).</summary>
    protected virtual Tensor<T> ToDiffusionSpace(Tensor<T> waveform) => waveform;

    /// <summary>The waveform <c>[1, 1, samples]</c> of a diffusion-space sample; the inverse of
    /// <see cref="ToDiffusionSpace"/>.</summary>
    protected virtual Tensor<T> FromDiffusionSpace(Tensor<T> sample) => sample;

    /// <summary>The training loss between the true and predicted noise; the default is the mean squared error
    /// (DDPM's simplified objective, ‖ε − ε_θ‖²).</summary>
    protected virtual Tensor<T> NoiseLoss(Tensor<T> noise, Tensor<T> predicted, Tensor<T> mel)
    {
        var d = Engine.TensorSubtract(noise, predicted);
        return Mean(Engine.TensorMultiply(d, d));
    }

    /// <summary>Whether each sampling step clamps the sample to [−1, 1] (the DiffWave, WaveGrad, PriorGrad and FreGrad
    /// references do).</summary>
    protected virtual bool ClampEachStep => false;

    // ---------------------------------------------------------------- helpers

    /// <summary>ᾱ_t = Π (1 − β_s) for every step.</summary>
    protected static double[] Cumulative(double[] betas)
    {
        var result = new double[betas.Length];
        double product = 1;
        for (int i = 0; i < betas.Length; i++)
        {
            product *= 1 - betas[i];
            result[i] = product;
        }
        return result;
    }

    /// <summary>A log10 magnitude mel spectrogram as <c>clamp((20 · log10 − 20 + 100) / 100, 0, 1)</c>, the feature
    /// scaling of the lmnt-com DiffWave and WaveGrad references (<c>preprocess.py</c>).</summary>
    protected Tensor<T> DecibelUnitRange(Tensor<T> log10)
    {
        // (20 · log10 − 20 + 100) / 100 = 0.2 · log10 + 0.8, clamped to [0, 1].
        var scaled = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(log10, NumOps.FromDouble(0.2)), NumOps.FromDouble(0.8));
        var upper = Engine.TensorAddScalar(Engine.TensorNegate(Engine.ReLU(Engine.TensorAddScalar(Engine.TensorNegate(scaled), NumOps.One))), NumOps.One);
        return Engine.ReLU(upper);
    }

    // ---------------------------------------------------------------- the process

    /// <inheritdoc />
    /// <remarks>Ancestral sampling over the inference schedule (or the training one) from the prior noise.</remarks>
    protected override Tensor<T> Synthesize(Tensor<T> mel, Random random)
    {
        int samples = mel.Shape[2] * UpsampleFactor;
        var betas = InferenceBetas ?? TrainingBetas;
        var alphaBar = Cumulative(betas);
        var x = PriorNoise(mel, samples, random);
        for (int n = betas.Length - 1; n >= 0; n--)
        {
            double alpha = 1 - betas[n];
            var eps = Denoise(x, InferenceLevel(n, alphaBar[n]), mel);
            x = Engine.TensorMultiplyScalar(Engine.TensorSubtract(x, Engine.TensorMultiplyScalar(eps, NumOps.FromDouble(betas[n] / Math.Sqrt(1 - alphaBar[n])))),
                NumOps.FromDouble(1 / Math.Sqrt(alpha)));
            if (n > 0)
            {
                double sigma = Math.Sqrt((1 - alphaBar[n - 1]) / (1 - alphaBar[n]) * betas[n]);
                x = Engine.TensorAdd(x, Engine.TensorMultiplyScalar(PriorNoise(mel, samples, random), NumOps.FromDouble(sigma)));
            }
            if (ClampEachStep)
                x = Engine.TensorAddScalar(Engine.ReLU(Engine.TensorAddScalar(Engine.TensorNegate(Engine.ReLU(Engine.TensorAddScalar(Engine.TensorNegate(x),
                    NumOps.One))), NumOps.FromDouble(2))), NumOps.FromDouble(-1));                                    // clamp(x, −1, 1)
        }
        return FromDiffusionSpace(x);
    }

    /// <inheritdoc />
    /// <remarks>One draw of the level and the prior noise, and the model's noise loss at it.</remarks>
    protected override Tensor<T> TrainingObjective(Tensor<T> mel, Tensor<T> audio, Random random)
    {
        var (level, sqrtAlphaBar) = DrawTrainingLevel(random);
        var noise = PriorNoise(mel, audio.Length, random);
        var x0 = ToDiffusionSpace(Engine.Reshape(audio, new[] { 1, 1, audio.Length }));
        var noisy = Engine.TensorAdd(Engine.TensorMultiplyScalar(x0, NumOps.FromDouble(sqrtAlphaBar)),
            Engine.TensorMultiplyScalar(noise, NumOps.FromDouble(Math.Sqrt(Math.Max(0, 1 - sqrtAlphaBar * sqrtAlphaBar)))));
        return NoiseLoss(noise, Denoise(noisy, level, mel), mel);
    }
}
