namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Sandbox configuration for executing untrusted code.
/// </summary>
public sealed class ServingSandboxOptions
{
    public ServingSandboxLimitOptions Free { get; set; } = new()
    {
        TimeLimitSeconds = 3,
        MemoryLimitMb = 128,
        CpuLimit = 0.5,
        MaxSourceCodeChars = 50_000,
        MaxStdInChars = 10_000,
        MaxStdOutChars = 16_000,
        MaxStdErrChars = 16_000,
        MaxConcurrentExecutions = 2
    };

    public ServingSandboxLimitOptions Premium { get; set; } = new()
    {
        TimeLimitSeconds = 10,
        MemoryLimitMb = 512,
        CpuLimit = 1.0,
        MaxSourceCodeChars = 200_000,
        MaxStdInChars = 100_000,
        MaxStdOutChars = 64_000,
        MaxStdErrChars = 64_000,
        MaxConcurrentExecutions = 8
    };

    public ServingSandboxLimitOptions Enterprise { get; set; } = new()
    {
        TimeLimitSeconds = 30,
        MemoryLimitMb = 1024,
        CpuLimit = 2.0,
        MaxSourceCodeChars = 1_000_000,
        MaxStdInChars = 250_000,
        MaxStdOutChars = 256_000,
        MaxStdErrChars = 256_000,
        MaxConcurrentExecutions = 16
    };
}
