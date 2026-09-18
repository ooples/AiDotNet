using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Tasks.Graph;

/// <summary>
/// Configuration options for <see cref="GraphClassificationModel{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Graph classification labels a whole graph rather than its parts — is
/// this molecule toxic, is this transaction network fraudulent. The values here are the ones the
/// model ships with.
/// </para>
/// </remarks>
public class GraphClassificationOptions : GraphModelOptions
{
    /// <summary>
    /// Initializes a new instance carrying the model's published defaults.
    /// </summary>
    public GraphClassificationOptions()
    {
        HiddenDim = 64;
        EmbeddingDim = 128;
        NumGnnLayers = 3;
        DropoutRate = 0.5;
        PoolingType = GraphPooling.Mean;
    }

    /// <summary>
    /// Initializes a new instance by copying another instance.
    /// </summary>
    /// <param name="other">The instance to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public GraphClassificationOptions(GraphClassificationOptions other)
    {
        if (other is null)
        {
            throw new ArgumentNullException(nameof(other));
        }

        Seed = other.Seed;
        HiddenDim = other.HiddenDim;
        EmbeddingDim = other.EmbeddingDim;
        NumGnnLayers = other.NumGnnLayers;
        DropoutRate = other.DropoutRate;
        PoolingType = other.PoolingType;
        MaxGradNorm = other.MaxGradNorm;
    }

    /// <summary>
    /// Gets or sets the width of the pooled whole-graph embedding. Default: 128.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After the model summarises every node, the whole graph is
    /// squeezed into a list of this many numbers, and the label is predicted from that.</para>
    /// </remarks>
    public int EmbeddingDim { get; set; }

    /// <summary>
    /// Gets or sets the number of message-passing layers. Default: 3.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Named NumGnnLayers rather than NumLayers because that is the name this model publishes;
    /// the node and link heads call the same idea NumLayers.
    /// </para>
    /// </remarks>
    public int NumGnnLayers { get; set; }

    /// <summary>
    /// Gets or sets how per-node representations are combined into one per-graph representation.
    /// Default: <see cref="GraphPooling.Mean"/>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The model produces one summary per node and needs a single
    /// summary for the whole graph. Mean averages them; other choices take the maximum or the
    /// sum, which emphasise different things.</para>
    /// </remarks>
    public GraphPooling PoolingType { get; set; }

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
        Require(NumGnnLayers, nameof(NumGnnLayers));
    }
}
