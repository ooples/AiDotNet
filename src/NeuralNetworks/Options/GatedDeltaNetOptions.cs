using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the GatedDeltaNetLanguageModel.
/// </summary>
public class GatedDeltaNetOptions : NeuralNetworkOptions
{
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
}
