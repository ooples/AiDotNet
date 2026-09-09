using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the InfographicVQA document model.
/// </summary>
public class InfographicVQAOptions : DocumentNeuralNetworkOptions
{
    /// <summary>Initializes an options instance with default values.</summary>
    public InfographicVQAOptions()
    {
        ImageSize = 1024;
        MaxSequenceLength = 512;
        VisionDim = 768;
        TextDim = 768;
        FusionDim = 768;
        VisionLayers = 12;
        FusionLayers = 6;
        NumHeads = 12;
        VocabSize = 30522;
    }

    /// <summary>Initializes an options instance by copying inherited configuration.</summary>
    /// <param name="other">The source options.</param>
    public InfographicVQAOptions(InfographicVQAOptions other)
    {
        if (other is null)
            throw new ArgumentNullException(nameof(other));
        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        ImageSize = other.ImageSize;
        MaxSequenceLength = other.MaxSequenceLength;
        VisionDim = other.VisionDim;
        VisionLayers = other.VisionLayers;
        NumHeads = other.NumHeads;
        VocabSize = other.VocabSize;
        TextDim = other.TextDim;
        FusionDim = other.FusionDim;
        FusionLayers = other.FusionLayers;
    }

    /// <summary>
    /// Gets or sets the text dim.
    /// </summary>
    public int TextDim { get; set; }

    /// <summary>
    /// Gets or sets the fusion dim.
    /// </summary>
    public int FusionDim { get; set; }

    /// <summary>
    /// Gets or sets the fusion layers.
    /// </summary>
    public int FusionLayers { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore();
    }
}
