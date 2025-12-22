using AiDotNet.Serving.Scheduling;

namespace AiDotNet.Serving.Services;

/// <summary>
/// Represents a queued request for the <see cref="ContinuousBatchingRequestBatcher"/>.
/// </summary>
internal sealed class ContinuousRequest
{
    public long RequestId { get; set; }
    public string ModelName { get; set; } = string.Empty;
    public string NumericType { get; set; } = string.Empty;
    public object Input { get; set; } = null!;
    public object CompletionSource { get; set; } = null!;
    public RequestPriority Priority { get; set; } = RequestPriority.Normal;
    public DateTime EnqueueTime { get; set; }
}

