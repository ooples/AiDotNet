using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Tasks.Graph;

/// <summary>
/// Configuration options for <see cref="LinkPredictionModel{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Link prediction guesses which connections a graph is missing — who else
/// you might know, which molecule might bind to which protein. The values here are the ones the
/// model ships with.
/// </para>
/// </remarks>
public class LinkPredictionOptions : GraphModelOptions
{
    /// <summary>
    /// Initializes a new instance carrying the model's published defaults.
    /// </summary>
    public LinkPredictionOptions()
    {
        HiddenDim = 64;
        EmbeddingDim = 32;
        NumLayers = 2;
        DropoutRate = 0.5;
        DecoderType = LinkPredictionDecoder.DotProduct;
    }

    /// <summary>
    /// Initializes a new instance by copying another instance.
    /// </summary>
    /// <param name="other">The instance to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public LinkPredictionOptions(LinkPredictionOptions other)
    {
        if (other is null)
        {
            throw new ArgumentNullException(nameof(other));
        }

        Seed = other.Seed;
        HiddenDim = other.HiddenDim;
        EmbeddingDim = other.EmbeddingDim;
        NumLayers = other.NumLayers;
        DropoutRate = other.DropoutRate;
        DecoderType = other.DecoderType;
        MaxGradNorm = other.MaxGradNorm;
    }

    /// <summary>
    /// Gets or sets the width of the final node embedding used to score a candidate edge.
    /// Default: 32.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each node ends up as a list of this many numbers, and two nodes
    /// are scored as a likely connection by comparing their lists.</para>
    /// </remarks>
    public int EmbeddingDim { get; set; }

    /// <summary>
    /// Gets or sets the number of message-passing layers. Default: 2.
    /// </summary>
    public int NumLayers { get; set; }

    /// <summary>
    /// Gets or sets how a candidate edge is scored from its two node embeddings.
    /// Default: <see cref="LinkPredictionDecoder.DotProduct"/>.
    /// </summary>
    public LinkPredictionDecoder DecoderType { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore();
        Require(EmbeddingDim, nameof(EmbeddingDim));
        Require(NumLayers, nameof(NumLayers));
    }
}
