using System.Threading;
using System.Threading.Tasks;

namespace AiDotNet.Serving.Benchmarks.Backends;

/// <summary>
/// Adapter over a serving endpoint. Implementations translate a <see cref="RequestSpec"/> into the
/// backend's wire protocol, measure timing, and return a <see cref="RequestResult"/>. The load
/// runner supplies <paramref name="dispatchMs"/> (time of dispatch relative to run start) so all
/// per-request timings share a common clock origin.
/// </summary>
public interface IServingBackend
{
    /// <summary>Short adapter name for reporting.</summary>
    string Name { get; }

    /// <summary>Executes one request end-to-end and returns its measurements.</summary>
    Task<RequestResult> ExecuteAsync(RequestSpec spec, double dispatchMs, CancellationToken ct);
}
