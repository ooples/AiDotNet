namespace AiDotNet.SelfSupervisedLearning.Core;

/// <summary>
/// MoCo-specific configuration settings.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> MoCo (Momentum Contrast) uses a memory queue and momentum encoder
/// to provide consistent negative samples without large batch sizes.</para>
/// </remarks>
public class MoCoConfig
{
    /// <summary>
    /// Gets or sets the size of the memory queue.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>65536</c></para>
    /// <para>Larger queues provide more negative samples.</para>
    /// </remarks>
    public int? QueueSize { get; set; }

    /// <summary>
    /// Gets or sets the momentum coefficient for the momentum encoder.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.999</c></para>
    /// <para>Higher momentum = slower updates = more stable negatives.</para>
    /// </remarks>
    public double? Momentum { get; set; }

    /// <summary>
    /// Gets or sets whether to use MLP projection head (MoCo v2+).
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>true</c></para>
    /// </remarks>
    public bool? UseMLPProjector { get; set; }

    /// <summary>
    /// Gets the configuration as a dictionary.
    /// </summary>
    public IDictionary<string, object> GetConfiguration()
    {
        var config = new Dictionary<string, object>();
        if (QueueSize.HasValue) config["queueSize"] = QueueSize.Value;
        if (Momentum.HasValue) config["momentum"] = Momentum.Value;
        if (UseMLPProjector.HasValue) config["useMLPProjector"] = UseMLPProjector.Value;
        return config;
    }
}
