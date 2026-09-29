namespace AiDotNet.Models.Options;

/// <summary>
/// Options shared by the synthetic tabular data generators — the GAN, VAE and diffusion models in
/// <c>src/NeuralNetworks/SyntheticData</c> that learn a data distribution and sample new rows from
/// it.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These models generate believable fake records that look like your real
/// data — useful when the real data cannot be shared. They train two or more networks against each
/// other, which is a less stable process than ordinary supervised training, and that instability is
/// what this class exists to account for.
/// </para>
/// <para>
/// <b>Why this sits between the generators and <see cref="RiskModelOptions{T}"/>.</b> All 21 of
/// these generators declared a gradient-clipping threshold of 5.0, while the twelve tabular
/// <i>prediction</i> models that share <see cref="RiskModelOptions{T}"/> as a base declared 1.0.
/// They were siblings with different values, so neither number could live on the shared parent:
/// putting 5.0 there would have loosened clipping on the twelve, and leaving the inherited 1.0
/// would have tightened it on these 21. The split is exactly the
/// <c>SyntheticData</c> / <c>Tabular</c> folder boundary, so the family is a real one rather than
/// an artefact of the migration.
/// </para>
/// </remarks>
public class SyntheticDataGeneratorOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Initializes a new instance carrying the defaults shared by the synthetic data generators.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Adversarial training produces much larger and spikier gradients than supervised training —
    /// a discriminator that suddenly wins sends a very large signal back to the generator — so
    /// clipping at the library's default of 1.0 would cap ordinary updates rather than only the
    /// destructive ones. 5.0 is the value every one of these models already used, preserved here
    /// so behaviour is unchanged.
    /// </para>
    /// </remarks>
    public SyntheticDataGeneratorOptions()
    {
        MaxGradNorm = 5.0;
    }
}
