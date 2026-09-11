using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the GLALanguageModel.
/// </summary>
public class GLAOptions : SequenceModelOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public GLAOptions()
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
    /// Copies <see cref="LearningRate"/> here and delegates inherited settings to the base copy
    /// constructor. Callers can derive a new configuration without silently reverting any value
    /// to its default. This options-copy contract is independent of the generated model clone plan.
    /// </remarks>
    public GLAOptions(GLAOptions other) : base(other)
    {
        LearningRate = other.LearningRate;
    }

    /// <summary>
    /// Gets or sets the peak AdamW learning rate used when the model builds its own optimizer.
    /// </summary>
    /// <value>Defaults to 3e-4, the rate the GLA paper pretrains with (arXiv:2312.06635).</value>
    /// <remarks>
    /// <para>
    /// The model previously constructed <c>AdamWOptimizer</c> with no options at all, so it trained at
    /// the library-wide AdamW default of 1e-3 -- neither the published rate nor reachable by a caller
    /// who passed <c>options</c> but let the optimizer default. A supplied optimizer still controls
    /// the actual training rate; the stored options must nevertheless pass <see cref="Validate"/>,
    /// including a finite, positive learning rate, before model construction.
    /// </para>
    /// <para><b>For Beginners:</b> How big a step the model takes each time it learns. The default is the
    /// value the paper's authors used, so training here starts from the same recipe they published.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 3e-4;

    /// <summary>
    /// Throws if a required model dimension or consumed training setting is invalid.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is non-positive, or a consumed numeric setting is
    /// non-finite or outside its supported range. The message identifies the invalid property.
    /// </exception>
    public void Validate()
    {
        ValidateCore(requiresHeads: true, requiresState: false);
        Require(LearningRate, nameof(LearningRate));
    }
}
