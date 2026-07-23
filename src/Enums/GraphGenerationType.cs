namespace AiDotNet.Enums;

/// <summary>
/// Type of graph generation approach.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These are different strategies for generating new graphs:
///
/// - **VariationalAutoencoder**: Learns a latent space representation and generates by sampling
/// - **Autoregressive**: Generates graphs one node/edge at a time sequentially
/// - **OneShot**: Generates the entire graph structure in a single pass
/// </para>
/// </remarks>
public enum GraphGenerationType
{
    /// <summary>
    /// Variational autoencoder approach - learns latent space for generation.
    /// </summary>
    VariationalAutoencoder = 0,

    /// <summary>
    /// Autoregressive generation - generates nodes/edges sequentially.
    /// </summary>
    Autoregressive = 1,

    /// <summary>
    /// One-shot generation - generates entire graph at once.
    /// </summary>
    OneShot = 2
}
