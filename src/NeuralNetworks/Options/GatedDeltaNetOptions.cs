using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the GatedDeltaNetLanguageModel.
/// </summary>
public class GatedDeltaNetOptions : SequenceModelOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public GatedDeltaNetOptions()
    {
        VocabSize = 50277;
        ModelDimension = 256;
        NumLayers = 4;
        NumHeads = 8;
        MaxSequenceLength = 512;
    }

    /// <summary>
    /// Initializes a new instance by copying every property from another instance.
    /// </summary>
    /// <param name="other">The instance to copy from.</param>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="other"/> is null.
    /// </exception>
    /// <remarks>
    /// <see cref="LearningRate"/> IS COPIED HERE, AND MUST STAY COPIED. This constructor is what
    /// <c>GatedDeltaNetLanguageModel&lt;T&gt;.CreateNewInstance</c> calls, so a property
    /// missing from it is not merely absent from the clone -- the clone silently reverts to the
    /// default while the original keeps the configured value, and nothing reports the divergence.
    /// A model cloned for evaluation would then train at a different rate than the one it was
    /// cloned from.
    /// </remarks>
    public GatedDeltaNetOptions(GatedDeltaNetOptions other)
    {
        if (other is null)
            throw new ArgumentNullException(nameof(other));
            Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        LearningRate = other.LearningRate;
        VocabSize = other.VocabSize;
        ModelDimension = other.ModelDimension;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        MaxSequenceLength = other.MaxSequenceLength;
    }

    /// <summary>
    /// Gets or sets the peak AdamW learning rate used when the model builds its own optimizer.
    /// </summary>
    /// <value>Defaults to 3e-4, the rate the Gated DeltaNet paper pretrains with (arXiv:2412.06464).</value>
    /// <remarks>
    /// <para>
    /// The model previously constructed <c>AdamWOptimizer</c> with no options at all, so it trained at
    /// the library-wide AdamW default of 1e-3 -- neither the published rate nor reachable by a caller
    /// who passed <c>options</c> but let the optimizer default. Supplying your own optimizer still wins;
    /// this value is only consulted when the model has to build one.
    /// </para>
    /// <para><b>For Beginners:</b> How big a step the model takes each time it learns. The default is the
    /// value the paper's authors used, so training here starts from the same recipe they published.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 3e-4;

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore(requiresHeads: true, requiresState: false);
    }
}
