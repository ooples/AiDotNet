using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the VisionTransformer.
/// </summary>
public class VisionTransformerOptions : ModelHyperparameterOptions
{

    /// <summary>Initializes a new instance carrying the model's published defaults.</summary>
    /// <remarks>
    /// <para>
    /// ViT-Base/16 (Dosovitskiy et al. 2021): a 768-wide encoder, 12 layers, 12 attention heads
    /// and a 3072-wide MLP.
    /// </para>
    /// </remarks>
    public VisionTransformerOptions()
    {
        HiddenDim = 768;
        NumLayers = 12;
        NumHeads = 12;
        MlpDim = 3072;
    }

    /// <summary>Gets or sets the width of the transformer encoder. Default: 768.</summary>
    /// <remarks>
    /// <para>
    /// Must be divisible by <see cref="NumHeads"/>; multi-head attention splits this width
    /// evenly between the heads.
    /// </para>
    /// </remarks>
    public int HiddenDim { get; set; }

    /// <summary>Gets or sets the number of transformer blocks. Default: 12.</summary>
    public int NumLayers { get; set; }

    /// <summary>Gets or sets the number of attention heads. Default: 12.</summary>
    public int NumHeads { get; set; }

    /// <summary>Gets or sets the width of the feed-forward block. Default: 3072.</summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each transformer block widens the representation, applies a
    /// non-linearity and narrows it again. This is that wider width, conventionally four times
    /// <see cref="HiddenDim"/>.</para>
    /// </remarks>
    public int MlpDim { get; set; }

    /// <summary>Throws if a value this model requires has been left unset or is not positive.</summary>
    /// <exception cref="ArgumentException">Thrown when a required dimension is zero or negative.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <see cref="HiddenDim"/> is not divisible by <see cref="NumHeads"/>.
    /// </exception>
    public void Validate()
    {
        Require(HiddenDim, nameof(HiddenDim));
        Require(NumLayers, nameof(NumLayers));
        Require(NumHeads, nameof(NumHeads));
        Require(MlpDim, nameof(MlpDim));

        // Cross-field: multi-head attention splits HiddenDim evenly between the heads, so an
        // indivisible pair describes an architecture that cannot be built. A range relationship
        // rather than an unset value, so it keeps ArgumentOutOfRangeException.
        if (HiddenDim % NumHeads != 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(HiddenDim),
                $"HiddenDim {HiddenDim} must be divisible by NumHeads {NumHeads} for multi-head attention.");
        }
    }
}
