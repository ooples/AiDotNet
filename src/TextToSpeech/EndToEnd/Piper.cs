using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>Piper: a fast, local neural text-to-speech system — VITS trained with Rhasspy's recipe and model sizes.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> rhasspy/piper (<c>piper_train</c>, 2023), which trains VITS (Kim et al. 2021) unchanged;
/// see <see cref="PiperOptions"/> for the sizes and framing it sets.</para>
/// <para>
/// The network, its training and its synthesis are VITS's (<see cref="VITS{T}"/>, with the same stochastic duration
/// predictor). Piper reads phoneme ids framed as
/// piper-phonemize frames them — beginning-of-sentence, then every id followed by the padding id, then end-of-sentence
/// (<c>^ _ p₁ _ p₂ _ … pₙ _ $</c>) — and its medium and x-low voices decode with a lighter HiFi-GAN.
/// </para>
/// <para><b>For Beginners:</b> Piper is a small, fast VITS voice meant to run on ordinary computers and devices; it turns
/// phonemes straight into audio.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Piper: A Fast Local Neural Text-to-Speech System",
    "https://github.com/rhasspy/piper"
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-4, Beta1 = 0.8, Beta2 = 0.99, Epsilon = 1e-9, WeightDecay = 0.01,
                DecayRate = 0.999875, ReferenceBatchSize = 32,
                Source = "rhasspy/piper piper_train: AdamW (betas 0.8, 0.99, eps 1e-9, PyTorch default weight decay 0.01) at "
                        + "2e-4 with ExponentialLR 0.999875 per epoch; TRAINING.md trains with batch size 32.")]
public partial class Piper<T> : StochasticDurationVitsBase<T>
{
    /// <summary>Creates a Piper model that runs an exported ONNX voice.</summary>
    public Piper(NeuralNetworkArchitecture<T> architecture, string modelPath, PiperOptions? options = null)
        : base(architecture, modelPath, options ?? new PiperOptions())
    {
    }

    /// <summary>Creates a trainable Piper model.</summary>
    public Piper(NeuralNetworkArchitecture<T> architecture, PiperOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new PiperOptions(), optimizer)
    {
    }

    private PiperOptions PaperOptions => (PiperOptions)VitsOptions;

    /// <inheritdoc />
    /// <remarks>piper-phonemize's framing: <c>^ _ p₁ _ … pₙ _ $</c> (the padding only when
    /// <see cref="PiperOptions.InterspersePad"/>).</remarks>
    protected override Tensor<T> PrepareTokens(Tensor<T> tokens)
    {
        var o = PaperOptions;
        int step = o.InterspersePad ? 2 : 1;
        var ids = new Tensor<T>(new[] { step * (tokens.Length + 1) + 1 });
        int k = 0;
        ids[k++] = NumOps.FromDouble(o.BosId);
        if (o.InterspersePad) ids[k++] = NumOps.FromDouble(o.PadId);
        for (int i = 0; i < tokens.Length; i++)
        {
            ids[k++] = tokens[i];
            if (o.InterspersePad) ids[k++] = NumOps.FromDouble(o.PadId);
        }
        ids[k] = NumOps.FromDouble(o.EosId);
        return ids;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = PaperOptions;
        var m = new ModelMetadata<T>
        {
            Name = IsOnnxMode ? "Piper-ONNX" : "Piper-Native",
            Description = "Piper: a fast, local neural text-to-speech system (VITS, Rhasspy)",
            FeatureCount = o.HiddenDim,
            Complexity = o.NumEncoderLayers + o.NumFlowSteps,
        };
        m.AdditionalInfo["Architecture"] = "Piper";
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        return m;
    }
}
