using AiDotNet.Serving.Security;

namespace AiDotNet.Serving.ProgramSynthesis;

/// <summary>
/// Provides per-tier concurrency limiting for Program Synthesis code task endpoints.
/// </summary>
public interface IServingProgramSynthesisConcurrencyLimiter
{
    Task<IDisposable> AcquireAsync(ServingTier tier, CancellationToken cancellationToken = default);
}

