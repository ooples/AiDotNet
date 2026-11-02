namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Per-tier sandbox limits for untrusted execution.
/// </summary>
public sealed class ServingSandboxLimitOptions
{
    /// <summary>Gets or sets the wall-clock time one execution may run before it is stopped, in seconds.</summary>
    /// <value>Must be positive (values below 1 run as 1 second). Default: 5.</value>
    public int TimeLimitSeconds { get; set; } = 5;

    /// <summary>Gets or sets the container memory limit of one execution, in megabytes (Docker <c>--memory</c>).</summary>
    /// <value>Must be positive; the container gets at least 32 MB. Default: 256.</value>
    public int MemoryLimitMb { get; set; } = 256;

    /// <summary>Gets or sets the number of CPUs one execution may use (Docker <c>--cpus</c>; fractions allowed).</summary>
    /// <value>Must be positive; the container gets at least 0.1. Default: 1.0.</value>
    public double CpuLimit { get; set; } = 1.0;

    /// <summary>Gets or sets the longest source code accepted, in characters; longer requests are rejected unrun.</summary>
    /// <value>Must be positive. Default: 200,000.</value>
    public int MaxSourceCodeChars { get; set; } = 200_000;

    /// <summary>Gets or sets the longest standard input accepted, in characters; longer requests are rejected unrun.</summary>
    /// <value>Must be zero or more. Default: 100,000.</value>
    public int MaxStdInChars { get; set; } = 100_000;

    /// <summary>
    /// Gets or sets how much standard output is kept, in characters; output past it is cut off and the result is
    /// marked truncated.
    /// </summary>
    /// <value>Must be zero or more. Default: 64,000.</value>
    public int MaxStdOutChars { get; set; } = 64_000;

    /// <summary>
    /// Gets or sets how much standard error is kept, in characters; output past it is cut off and the result is
    /// marked truncated.
    /// </summary>
    /// <value>Must be zero or more. Default: 64,000.</value>
    public int MaxStdErrChars { get; set; } = 64_000;

    /// <summary>Gets or sets how many executions of this tier may run at once; further requests wait for a slot.</summary>
    /// <value>Must be positive. Default: 4.</value>
    public int MaxConcurrentExecutions { get; set; } = 4;

    /// <summary>
    /// Throws when a limit cannot be enforced: a non-positive time, memory, CPU, source-size or concurrency limit, or
    /// a negative input or output size limit.
    /// </summary>
    /// <param name="tierName">The tier these limits belong to, used in the error message; blank reads as "Unknown".</param>
    /// <exception cref="InvalidOperationException">A limit is out of range.</exception>
    public void Validate(string tierName)
    {
        if (string.IsNullOrWhiteSpace(tierName))
        {
            tierName = "Unknown";
        }

        if (TimeLimitSeconds <= 0)
        {
            throw new InvalidOperationException($"{tierName}: TimeLimitSeconds must be > 0.");
        }

        if (MemoryLimitMb <= 0)
        {
            throw new InvalidOperationException($"{tierName}: MemoryLimitMb must be > 0.");
        }

        if (CpuLimit <= 0)
        {
            throw new InvalidOperationException($"{tierName}: CpuLimit must be > 0.");
        }

        if (MaxSourceCodeChars <= 0)
        {
            throw new InvalidOperationException($"{tierName}: MaxSourceCodeChars must be > 0.");
        }

        if (MaxStdInChars < 0)
        {
            throw new InvalidOperationException($"{tierName}: MaxStdInChars must be >= 0.");
        }

        if (MaxStdOutChars < 0)
        {
            throw new InvalidOperationException($"{tierName}: MaxStdOutChars must be >= 0.");
        }

        if (MaxStdErrChars < 0)
        {
            throw new InvalidOperationException($"{tierName}: MaxStdErrChars must be >= 0.");
        }

        if (MaxConcurrentExecutions <= 0)
        {
            throw new InvalidOperationException($"{tierName}: MaxConcurrentExecutions must be > 0.");
        }
    }
}
