namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Per-tier sandbox limits for untrusted execution.
/// </summary>
public sealed class ServingSandboxLimitOptions
{
    public int TimeLimitSeconds { get; set; } = 5;

    public int MemoryLimitMb { get; set; } = 256;

    public double CpuLimit { get; set; } = 1.0;

    public int MaxSourceCodeChars { get; set; } = 200_000;

    public int MaxStdInChars { get; set; } = 100_000;

    public int MaxStdOutChars { get; set; } = 64_000;

    public int MaxStdErrChars { get; set; } = 64_000;

    public int MaxConcurrentExecutions { get; set; } = 4;

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
