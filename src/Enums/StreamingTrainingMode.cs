namespace AiDotNet.Enums;

/// <summary>
/// Controls the memory-bounded streaming training path (optimizer-in-backward
/// with 8-bit Adam state and topological-min gradient release).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Very large models can run out of memory during
/// training because the gradients and optimizer state are several times the size
/// of the model itself. Streaming training applies each parameter's update the
/// instant its gradient is ready and then frees that gradient, so the full
/// gradient set never has to fit in memory at once. This setting decides when
/// that path is used.</para>
/// </remarks>
public enum StreamingTrainingMode
{
    /// <summary>
    /// Engage streaming automatically: the autotuner turns it on only when the
    /// model's estimated full-precision training footprint would not comfortably
    /// fit in available memory. Models that already fit train on the classic
    /// path with zero overhead and bit-identical results. This is the default.
    /// </summary>
    Auto = 0,

    /// <summary>Always use the streaming training path (mainly for tests).</summary>
    ForceOn = 1,

    /// <summary>Never use the streaming training path (classic in-memory training only).</summary>
    ForceOff = 2,
}
