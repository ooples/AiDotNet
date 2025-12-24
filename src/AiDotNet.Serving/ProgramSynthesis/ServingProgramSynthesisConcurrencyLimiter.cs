using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Security;
using Microsoft.Extensions.Options;

namespace AiDotNet.Serving.ProgramSynthesis;

/// <summary>
/// Default per-tier concurrency limiter for Program Synthesis code task endpoints.
/// </summary>
public sealed class ServingProgramSynthesisConcurrencyLimiter : IServingProgramSynthesisConcurrencyLimiter
{
    private readonly SemaphoreSlim _free;
    private readonly SemaphoreSlim _premium;
    private readonly SemaphoreSlim _enterprise;

    public ServingProgramSynthesisConcurrencyLimiter(IOptions<ServingProgramSynthesisOptions> options)
    {
        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        var o = options.Value ?? new ServingProgramSynthesisOptions();
        _free = new SemaphoreSlim(Math.Max(1, o.Free.MaxConcurrentRequests));
        _premium = new SemaphoreSlim(Math.Max(1, o.Premium.MaxConcurrentRequests));
        _enterprise = new SemaphoreSlim(Math.Max(1, o.Enterprise.MaxConcurrentRequests));
    }

    public async Task<IDisposable> AcquireAsync(ServingTier tier, CancellationToken cancellationToken = default)
    {
        var semaphore = GetSemaphore(tier);
        await semaphore.WaitAsync(cancellationToken).ConfigureAwait(false);
        return new Lease(semaphore);
    }

    private SemaphoreSlim GetSemaphore(ServingTier tier) =>
        tier switch
        {
            ServingTier.Premium => _premium,
            ServingTier.Enterprise => _enterprise,
            _ => _free
        };

    private sealed class Lease : IDisposable
    {
        private SemaphoreSlim? _semaphore;

        public Lease(SemaphoreSlim semaphore)
        {
            _semaphore = semaphore;
        }

        public void Dispose()
        {
            var s = Interlocked.Exchange(ref _semaphore, null);
            s?.Release();
        }
    }
}

