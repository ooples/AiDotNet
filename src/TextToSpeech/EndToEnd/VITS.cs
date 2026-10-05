using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>
/// VITS: end-to-end text-to-speech as a conditional VAE whose prior is a normalizing flow over a text encoding, with a
/// stochastic duration predictor and a HiFi-GAN decoder trained adversarially.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech"
/// (Kim et al., ICML 2021) and its reference implementation (jaywalnut310/vits) for what the paper leaves unstated.</para>
/// <para>
/// The posterior encoder maps the linear spectrogram to z ~ q(z | x); the flow f_θ maps z to the prior space, where the
/// text encoder's per-token N(μ, σ) is aligned to the frames by monotonic alignment search (§2.2.1). The stochastic
/// duration predictor learns a lower bound on the alignment's durations (§2.2.2) and trains with everything else. A random
/// 8192-sample window of z is decoded by the HiFi-GAN generator. The generator loss is the mel L1 × 45 + KL + duration +
/// adversarial + feature matching (Eq. 9); the period and scale discriminators train on the LSGAN loss (Eq. 7). Synthesis
/// samples durations (noise scale 0.8), expands the prior, draws z = μ + ε σ · 0.667, inverts the flow and decodes the
/// whole sequence.
/// </para>
/// <para><b>For Beginners:</b> VITS goes straight from text to a waveform in one network: it learns a hidden "voice
/// space" from real audio, learns to predict that space from text, and learns a vocoder that turns it into sound.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech",
    "https://arxiv.org/abs/2106.06103",
    Year = 2021,
    Authors = "Kim et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-4, Beta1 = 0.8, Beta2 = 0.99, Epsilon = 1e-9, WeightDecay = 0.01,
                DecayRate = 0.999875, ReferenceBatchSize = 64,
                Source = "Kim et al. 2021, Sec. 4.1: AdamW with beta1 0.8, beta2 0.99 and weight decay 0.01, an initial "
                        + "learning rate of 2e-4 decayed by 0.999^(1/8) every epoch, batch size 64.")]
public partial class VITS<T> : StochasticDurationVitsBase<T>
{
    /// <summary>Creates a VITS model that runs an exported ONNX graph.</summary>
    public VITS(NeuralNetworkArchitecture<T> architecture, string modelPath, VITSOptions? options = null)
        : base(architecture, modelPath, options ?? new VITSOptions())
    {
    }

    /// <summary>Creates a trainable VITS model.</summary>
    public VITS(NeuralNetworkArchitecture<T> architecture, VITSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new VITSOptions(), optimizer)
    {
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = VitsOptions;
        var m = new ModelMetadata<T>
        {
            Name = IsOnnxMode ? "VITS-ONNX" : "VITS-Native",
            Description = "VITS: Conditional VAE with Adversarial Learning for End-to-End TTS (Kim et al., 2021)",
            FeatureCount = o.HiddenDim,
            Complexity = o.NumEncoderLayers + o.NumFlowSteps,
        };
        m.AdditionalInfo["Architecture"] = "VITS";
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        return m;
    }
}
