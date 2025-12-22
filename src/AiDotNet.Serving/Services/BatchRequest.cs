using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Scheduling;

namespace AiDotNet.Serving.Services;

/// <summary>
/// Represents a queued request for the <see cref="RequestBatcher"/>.
/// </summary>
internal sealed class BatchRequest
{
    public string ModelName { get; set; } = string.Empty;
    public NumericType NumericType { get; set; } = NumericType.Double;
    public object Input { get; set; } = null!;
    public object CompletionSource { get; set; } = null!;
    public RequestPriority Priority { get; set; } = RequestPriority.Normal;
    public DateTime EnqueueTime { get; set; } = DateTime.UtcNow;
}
