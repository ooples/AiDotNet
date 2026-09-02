using System.Globalization;

namespace AiDotNet.Configuration;

/// <summary>Resource limits applied to every sandboxed execution of an untrusted candidate program.</summary>
/// <remarks>
/// <para>
/// The defaults mirror the free-tier limits that AiDotNet.Serving applies to container-sandboxed execution
/// (<c>ServingSandboxLimitOptions</c>): five seconds of wall clock, 256 MB of memory, one CPU, 200,000 source
/// characters, 100,000 standard-input characters, and 64,000 characters each of captured standard output and
/// standard error, with at most four executions running concurrently. Keeping the two sets of numbers identical
/// means a run that scores candidates locally and a run that scores them through a serving deployment agree on
/// which candidates time out, so a fitness landscape does not silently change shape when the sandbox does.
/// </para>
/// <para>
/// Every limit is enforced, not merely advertised. The time limit cancels the run and kills the whole process tree;
/// the memory limit becomes a Windows job-object cap or a <c>ulimit</c> on Unix; the output caps bound the reader
/// buffers, so a program that prints without stopping cannot exhaust the host's memory and is reported with the
/// truncation flag set; and the concurrency limit is a semaphore held for the duration of each execution.
/// </para>
/// <para><b>For Beginners:</b> These numbers are the leash on a program that a model wrote and nobody reviewed.
/// The time limit stops infinite loops, the memory limit stops runaway allocation, the output limits stop a program
/// that prints forever from filling your disk or your logs, and the concurrency limit stops a large population from
/// starting hundreds of processes at once. The defaults suit small algorithmic tasks; raise the time and memory
/// limits if your candidates legitimately need longer, and lower them if you want a tighter feedback loop.</para>
/// </remarks>
public sealed class ProgramSandboxLimitOptions
{
    /// <summary>The largest wall-clock limit accepted by <see cref="Validate"/>, in seconds.</summary>
    public const int MaxTimeLimitSeconds = 3600;

    /// <summary>The largest memory limit accepted by <see cref="Validate"/>, in megabytes.</summary>
    public const int MaxMemoryLimitMb = 1_048_576;

    /// <summary>The largest concurrency limit accepted by <see cref="Validate"/>.</summary>
    public const int MaxConcurrencyLimit = 4096;

    /// <summary>Gets or sets the wall-clock limit for one execution, in seconds. Defaults to 5.</summary>
    public int TimeLimitSeconds { get; set; } = 5;

    /// <summary>Gets or sets the memory limit for one execution, in megabytes. Defaults to 256.</summary>
    /// <remarks>
    /// Enforced through a Windows job object on Windows and through <c>ulimit -v</c> on Unix when a POSIX shell is
    /// available. Where neither mechanism is available the limit is not applied and the wall-clock limit remains the
    /// effective bound; the response never claims a limit was applied when it was not.
    /// </remarks>
    public int MemoryLimitMb { get; set; } = 256;

    /// <summary>Gets or sets the CPU-core allowance for one execution. Defaults to 1.0.</summary>
    /// <remarks>
    /// A process sandbox cannot cap CPU shares the way a container can, so this value is used to derive the CPU-time
    /// limit passed to <c>ulimit -t</c> on Unix and is reported to callers that forward the request to a
    /// container-backed engine. It never relaxes <see cref="TimeLimitSeconds"/>.
    /// </remarks>
    public double CpuLimit { get; set; } = 1.0;

    /// <summary>Gets or sets the largest accepted program source, in characters. Defaults to 200,000.</summary>
    public int MaxSourceCodeChars { get; set; } = 200_000;

    /// <summary>Gets or sets the largest accepted standard input, in characters. Defaults to 100,000.</summary>
    public int MaxStdInChars { get; set; } = 100_000;

    /// <summary>Gets or sets how many characters of standard output are captured. Defaults to 64,000.</summary>
    /// <remarks>Output beyond this bound is discarded and the response reports it as truncated.</remarks>
    public int MaxStdOutChars { get; set; } = 64_000;

    /// <summary>Gets or sets how many characters of standard error are captured. Defaults to 64,000.</summary>
    /// <remarks>Output beyond this bound is discarded and the response reports it as truncated.</remarks>
    public int MaxStdErrChars { get; set; } = 64_000;

    /// <summary>Gets or sets how many executions may run at the same time. Defaults to 4.</summary>
    public int MaxConcurrentExecutions { get; set; } = 4;

    /// <summary>Creates an independent copy so a running engine is unaffected by later mutation.</summary>
    /// <returns>A new instance carrying the same limits.</returns>
    public ProgramSandboxLimitOptions Clone() => new()
    {
        TimeLimitSeconds = TimeLimitSeconds,
        MemoryLimitMb = MemoryLimitMb,
        CpuLimit = CpuLimit,
        MaxSourceCodeChars = MaxSourceCodeChars,
        MaxStdInChars = MaxStdInChars,
        MaxStdOutChars = MaxStdOutChars,
        MaxStdErrChars = MaxStdErrChars,
        MaxConcurrentExecutions = MaxConcurrentExecutions
    };

    /// <summary>Rejects limits that cannot be enforced or that describe an impossible sandbox.</summary>
    /// <exception cref="ArgumentOutOfRangeException">
    /// A limit is not positive where a positive value is required, is negative where a non-negative value is
    /// required, is not a finite number, or exceeds the corresponding hard ceiling.
    /// </exception>
    public void Validate()
    {
        Require(TimeLimitSeconds > 0 && TimeLimitSeconds <= MaxTimeLimitSeconds, nameof(TimeLimitSeconds),
            TimeLimitSeconds, $"Value must be between 1 and {MaxTimeLimitSeconds} seconds.");
        Require(MemoryLimitMb > 0 && MemoryLimitMb <= MaxMemoryLimitMb, nameof(MemoryLimitMb),
            MemoryLimitMb, $"Value must be between 1 and {MaxMemoryLimitMb} megabytes.");
        Require(!double.IsNaN(CpuLimit) && !double.IsInfinity(CpuLimit) && CpuLimit > 0.0, nameof(CpuLimit),
            CpuLimit, "Value must be a positive finite number of CPU cores.");
        Require(MaxSourceCodeChars > 0, nameof(MaxSourceCodeChars),
            MaxSourceCodeChars, "Value must be greater than zero.");
        Require(MaxStdInChars >= 0, nameof(MaxStdInChars), MaxStdInChars, "Value cannot be negative.");
        Require(MaxStdOutChars >= 0, nameof(MaxStdOutChars), MaxStdOutChars, "Value cannot be negative.");
        Require(MaxStdErrChars >= 0, nameof(MaxStdErrChars), MaxStdErrChars, "Value cannot be negative.");
        Require(MaxConcurrentExecutions > 0 && MaxConcurrentExecutions <= MaxConcurrencyLimit,
            nameof(MaxConcurrentExecutions), MaxConcurrentExecutions,
            $"Value must be between 1 and {MaxConcurrencyLimit}.");
    }

    /// <summary>Gets the wall-clock limit as a time span.</summary>
    /// <returns>A positive time span derived from <see cref="TimeLimitSeconds"/>.</returns>
    public TimeSpan GetTimeLimit() =>
        TimeSpan.FromSeconds(TimeLimitSeconds > 0 ? TimeLimitSeconds : 1);

    /// <summary>Gets the memory limit in bytes.</summary>
    /// <returns>The memory limit converted from megabytes, or zero when the limit is not positive.</returns>
    public long GetMemoryLimitBytes() =>
        MemoryLimitMb > 0 ? (long)MemoryLimitMb * 1024L * 1024L : 0L;

    private static void Require(bool condition, string name, object value, string message)
    {
        if (!condition)
        {
            throw new ArgumentOutOfRangeException(
                name,
                value,
                string.Format(CultureInfo.InvariantCulture, "{0}: {1}", name, message));
        }
    }
}
