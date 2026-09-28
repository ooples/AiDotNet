using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the GraphGenerationModel.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This model learns to produce new graphs that look like the ones it was
/// trained on — new molecules, new networks. The values here are the ones it ships with, so you
/// can use it without configuring anything.
/// </para>
/// <para>
/// Derives from <see cref="ModelHyperparameterOptions"/> rather than one of the graph family
/// bases. It is a generator, not an encoder or a task head: it shares neither
/// <c>GraphEncoderOptions.NumLayers</c>'s meaning nor <c>GraphModelOptions.HiddenDim</c>'s
/// required-ness with those families, and its architecture is built from these values rather
/// than supplied.
/// </para>
/// </remarks>
public class GraphGenerationModelOptions : ModelHyperparameterOptions
{
    /// <summary>Initializes a new instance carrying the model's published defaults.</summary>
    public GraphGenerationModelOptions()
    {
        InputFeatures = 16;
        HiddenDim = 32;
        LatentDim = 16;
        NumEncoderLayers = 2;
        MaxNodes = 100;
        GenerationType = GraphGenerationType.VariationalAutoencoder;
        KlWeight = 1.0;
        LearningRate = 0.01;
        UseAMSGrad = false;
    }

    /// <summary>Initializes a new instance by copying another instance.</summary>
    /// <param name="other">The instance to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public GraphGenerationModelOptions(GraphGenerationModelOptions other)
    {
        if (other is null)
        {
            throw new ArgumentNullException(nameof(other));
        }

        Seed = other.Seed;
        InputFeatures = other.InputFeatures;
        HiddenDim = other.HiddenDim;
        LatentDim = other.LatentDim;
        NumEncoderLayers = other.NumEncoderLayers;
        MaxNodes = other.MaxNodes;
        GenerationType = other.GenerationType;
        KlWeight = other.KlWeight;
        LearningRate = other.LearningRate;
        UseAMSGrad = other.UseAMSGrad;
        MaxGradNorm = other.MaxGradNorm;
    }

    /// <summary>
    /// Gets or sets the number of features on each input node. Default: 16.
    /// </summary>
    public int InputFeatures { get; set; }

    /// <summary>
    /// Gets or sets the width of the encoder's hidden layers. Default: 32.
    /// </summary>
    public int HiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the width of the latent space the model samples new graphs from.
    /// Default: 16.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The model compresses a graph down to this many numbers, then
    /// generates new graphs by picking new numbers and expanding them back out.</para>
    /// </remarks>
    public int LatentDim { get; set; }

    /// <summary>
    /// Gets or sets the number of encoder layers. Default: 2.
    /// </summary>
    public int NumEncoderLayers { get; set; }

    /// <summary>
    /// Gets or sets the largest graph the model will generate, in nodes. Default: 100.
    /// </summary>
    public int MaxNodes { get; set; }

    /// <summary>
    /// Gets or sets the generation strategy.
    /// Default: <see cref="GraphGenerationType.VariationalAutoencoder"/>.
    /// </summary>
    public GraphGenerationType GenerationType { get; set; }

    /// <summary>
    /// Gets or sets the weight on the KL divergence term. Default: 1.0.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Balances two goals — reproducing the training graphs faithfully,
    /// and keeping the latent space tidy enough that new samples drawn from it are plausible.
    /// Zero is meaningful (it disables the term entirely), so this is not required positive.</para>
    /// </remarks>
    public double KlWeight { get; set; }

    /// <summary>
    /// Gets or sets the optimizer learning rate. Default: 0.01.
    /// </summary>
    public double LearningRate { get; set; }

    /// <summary>
    /// Gets or sets whether the optimizer uses the AMSGrad variant. Default: false.
    /// </summary>
    public bool UseAMSGrad { get; set; }

    /// <summary>Throws if a value this model requires has been left unset or is not positive.</summary>
    /// <exception cref="ArgumentException">Thrown when a required dimension is zero or negative.</exception>
    /// <remarks>
    /// <para>
    /// KlWeight and UseAMSGrad are deliberately not required: zero disables the KL term, which is
    /// a legitimate configuration.
    /// </para>
    /// </remarks>
    public void Validate()
    {
        Require(InputFeatures, nameof(InputFeatures));
        Require(HiddenDim, nameof(HiddenDim));
        Require(LatentDim, nameof(LatentDim));
        Require(NumEncoderLayers, nameof(NumEncoderLayers));
        Require(MaxNodes, nameof(MaxNodes));
        Require(LearningRate, nameof(LearningRate));
    }
}
