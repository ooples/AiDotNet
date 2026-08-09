using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the FinchLanguageModel.
/// </summary>
public class FinchOptions : NeuralNetworkOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public FinchOptions() { }

    /// <summary>Initializes a new instance by copying every property from another instance.</summary>
    /// <param name="other">The instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public FinchOptions(FinchOptions other)
    {
        if (other is null)
            throw new ArgumentNullException(nameof(other));
        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        LearningRate = other.LearningRate;
        MinLearningRate = other.MinLearningRate;
        Beta1 = other.Beta1;
        Beta2 = other.Beta2;
        WeightDecay = other.WeightDecay;
        EnableGradientClipping = other.EnableGradientClipping;
        MaxGradientNorm = other.MaxGradientNorm;
    }

    /// <summary>Gets or sets the maximum (peak) learning rate.</summary>
    /// <value>
    /// Defaults to 3e-4, the max learning rate the Finch paper trains its 1.6B model with
    /// (arXiv:2404.05892, Table 17).
    /// </value>
    /// <remarks>
    /// <para>
    /// The paper publishes the rate per model size rather than one universal value
    /// (Table 17: 0.4B 4e-4, 1.5B/1.6B 3e-4, 3B 2e-4, 7B 1.5e-4). The two models it actually calls
    /// Finch are 1.6B and 3.1B, so 3e-4 is the smaller of the two published Finch configurations.
    /// Set it to 2e-4 to reproduce the 3B run.
    /// </para>
    /// <para><b>For Beginners:</b> How big a step the model takes each time it learns.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 3e-4;

    /// <summary>Gets or sets the floor the cosine schedule decays to.</summary>
    /// <value>Defaults to 2e-5, the paper's minimum rate for the 1.6B model (Table 17).</value>
    /// <remarks>
    /// The paper's schedule is a linear 10-step warmup from 20% to 100% of the maximum rate,
    /// followed by cosine decay to this minimum (Appendix H).
    /// </remarks>
    public double MinLearningRate { get; set; } = 2e-5;

    /// <summary>Gets or sets Adam's first-moment decay.</summary>
    /// <value>Defaults to 0.9 (arXiv:2404.05892, Appendix H).</value>
    public double Beta1 { get; set; } = 0.9;

    /// <summary>Gets or sets Adam's second-moment decay.</summary>
    /// <value>Defaults to 0.99 (arXiv:2404.05892, Appendix H).</value>
    /// <remarks>
    /// NOTE the value: the paper uses 0.99, not the 0.999 that is conventional elsewhere in this
    /// library. It is not a typo -- copying a sibling model's 0.999 here would depart from the
    /// published recipe.
    /// </remarks>
    public double Beta2 { get; set; } = 0.99;

    /// <summary>Gets or sets the decoupled weight decay.</summary>
    /// <value>Defaults to 0.001 (arXiv:2404.05892, Appendix H).</value>
    /// <remarks>
    /// The paper applies decay only to linear layers and embedding weights. This value is the
    /// published magnitude; the per-parameter-group scoping is a property of the training loop
    /// rather than of this options object.
    /// </remarks>
    public double WeightDecay { get; set; } = 0.001;

    /// <summary>Gets or sets whether global-norm gradient clipping is enabled.</summary>
    /// <value>Defaults to <c>true</c>.</value>
    /// <remarks>
    /// NOT A PAPER VALUE. arXiv:2404.05892 states no gradient-clipping threshold, so this and
    /// <see cref="MaxGradientNorm"/> are a library safeguard rather than part of the published
    /// recipe -- RWKV-6's recurrent products can transiently overflow on an unscaled batch, which
    /// is what the model's previous hard-coded bound was defending against. They are called out
    /// separately so nobody later cites the paper for them.
    /// </remarks>
    public bool EnableGradientClipping { get; set; } = true;

    /// <summary>Gets or sets the maximum gradient norm. See <see cref="EnableGradientClipping"/>.</summary>
    /// <value>Defaults to 1.0. NOT a paper value -- see the remark on the property above.</value>
    public double MaxGradientNorm { get; set; } = 1.0;
}
