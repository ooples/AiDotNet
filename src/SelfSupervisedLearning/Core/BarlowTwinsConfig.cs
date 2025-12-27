namespace AiDotNet.SelfSupervisedLearning.Core;

/// <summary>
/// Barlow Twins-specific configuration settings.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Barlow Twins learns by making the cross-correlation matrix
/// between embeddings of two views close to the identity matrix (reducing redundancy).</para>
/// </remarks>
public class BarlowTwinsConfig
{
    /// <summary>
    /// Gets or sets the lambda parameter for redundancy reduction.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.0051</c></para>
    /// <para>Controls the weight of the off-diagonal terms in the loss.</para>
    /// </remarks>
    public double? Lambda { get; set; }

    /// <summary>
    /// Gets or sets the projection dimension.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>8192</c></para>
    /// <para>Barlow Twins uses much larger projection dimensions than other methods.</para>
    /// </remarks>
    public int? ProjectionDimension { get; set; }

    /// <summary>
    /// Gets the configuration as a dictionary.
    /// </summary>
    public IDictionary<string, object> GetConfiguration()
    {
        var config = new Dictionary<string, object>();
        if (Lambda.HasValue) config["lambda"] = Lambda.Value;
        if (ProjectionDimension.HasValue) config["projectionDimension"] = ProjectionDimension.Value;
        return config;
    }
}
