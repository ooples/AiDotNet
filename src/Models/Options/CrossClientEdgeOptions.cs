namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for handling cross-client edges in federated graph learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When a graph is split across clients, some edges connect nodes on
/// different clients. These "cross-client edges" are a unique challenge — we need to discover them
/// without revealing each client's full adjacency list. This class controls how that discovery works.</para>
///
/// <para><b>Options:</b></para>
/// <list type="bullet">
/// <item><description><b>PSI-based:</b> Use Private Set Intersection (#538) to discover shared edges
/// without revealing non-shared node IDs.</description></item>
/// <item><description><b>TEE-based:</b> Use a trusted enclave (#537) to compute edge intersections.</description></item>
/// <item><description><b>None:</b> Ignore cross-client edges entirely (simplest, lowest quality).</description></item>
/// </list>
/// </remarks>
public class CrossClientEdgeOptions
{
    /// <summary>
    /// Gets or sets whether to use PSI for edge discovery. Default is true.
    /// </summary>
    public bool UsePsi { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use TEE-based edge discovery as an alternative to PSI.
    /// Default is false.
    /// </summary>
    public bool UseTee { get; set; } = false;

    /// <summary>
    /// Gets or sets the differential privacy epsilon for edge queries. Default is 1.0.
    /// Lower values provide stronger privacy but may miss some cross-client edges.
    /// </summary>
    public double EdgePrivacyEpsilon { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the maximum number of cross-client edges to discover per client pair.
    /// Default is 1000. Set to 0 for unlimited.
    /// </summary>
    public int MaxEdgesPerClientPair { get; set; } = 1000;

    /// <summary>
    /// Gets or sets whether to cache discovered cross-client edges across rounds. Default is true.
    /// </summary>
    public bool CacheDiscoveredEdges { get; set; } = true;
}
