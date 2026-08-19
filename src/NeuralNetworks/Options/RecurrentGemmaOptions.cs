using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the RecurrentGemmaLanguageModel.
/// </summary>
/// <remarks>
/// <para>
/// Every value here is a PAPER DEFAULT assigned in the constructor rather than baked into a property
/// initializer, so a caller can replace any of them without having to reconstruct the rest. The
/// defaults reproduce RecurrentGemma (Botev et al., 2024); the type carried no settings at all before,
/// so the model silently inherited whatever the base supplied.
/// </para>
/// </remarks>
public class RecurrentGemmaOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Gets or sets whether the input embeddings are multiplied by the square root of the model width.
    /// </summary>
    /// <remarks>
    /// RecurrentGemma, Section 2: "multiply the input embeddings by a constant equal to the square root
    /// of model width." The paper notes the constant is NOT applied to the output, which is why this
    /// scales the embedding layer only and leaves the LM head alone.
    /// </remarks>
    public bool ScaleEmbeddingsBySqrtWidth { get; set; }

    // NOT EXPOSED: the paper's third measure, "we do not apply weight decay to the parameters of the
    // recurrent (RG-LRU) layers during training" (Section 2). AdamW here applies decay to the whole
    // flat parameter vector and has no per-parameter-group exclusion, so honouring it needs a decay
    // mask on the shared optimizer rather than a property on this type. A setting that silently did
    // nothing would be worse than its absence, so it is left out until the optimizer can enforce it.

    /// <summary>Gets or sets the AdamW learning rate.</summary>
    public double LearningRate { get; set; }

    /// <summary>Gets or sets the decoupled AdamW weight decay.</summary>
    public double WeightDecay { get; set; }

    /// <summary>Gets or sets AdamW's first-moment decay.</summary>
    public double Beta1 { get; set; }

    /// <summary>Gets or sets AdamW's second-moment decay.</summary>
    public double Beta2 { get; set; }

    /// <summary>Gets or sets AdamW's numerical-stability epsilon.</summary>
    public double Epsilon { get; set; }

    /// <summary>Gets or sets whether global-norm gradient clipping is enabled.</summary>
    public bool EnableGradientClipping { get; set; }

    /// <summary>Gets or sets the maximum gradient norm.</summary>
    public double MaxGradientNorm { get; set; }

    /// <summary>Initializes an options instance with the paper's defaults.</summary>
    public RecurrentGemmaOptions()
    {
        // RecurrentGemma, Section 2.
        ScaleEmbeddingsBySqrtWidth = true;

        // Shared with the Griffin/Hawk siblings, which build the same RG-LRU stack.
        LearningRate = 1e-4;
        WeightDecay = 0.01;
        Beta1 = 0.9;
        Beta2 = 0.999;
        Epsilon = 1e-8;
        EnableGradientClipping = true;
        MaxGradientNorm = 1.0;
    }

    /// <summary>Initializes an options instance by copying another.</summary>
    /// <param name="other">The source options.</param>
    public RecurrentGemmaOptions(RecurrentGemmaOptions other)
    {
        if (other is null)
            throw new ArgumentNullException(nameof(other));
        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        ScaleEmbeddingsBySqrtWidth = other.ScaleEmbeddingsBySqrtWidth;
        LearningRate = other.LearningRate;
        WeightDecay = other.WeightDecay;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        Epsilon = other.Epsilon;
        EnableGradientClipping = other.EnableGradientClipping;
        MaxGradientNorm = other.MaxGradientNorm;
    }
}
