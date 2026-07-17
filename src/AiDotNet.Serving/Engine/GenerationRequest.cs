namespace AiDotNet.Serving.Engine;

/// <summary>
/// A single generation request submitted to the inference engine: a tokenized prompt plus the sampling
/// parameters that govern its decoding. The engine owns scheduling, KV-cache allocation, and batching for
/// the request; the caller only supplies the prompt tokens (tokenization happens upstream) and consumes the
/// streamed <see cref="RequestOutput"/>s the engine produces.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> think of this as one "please generate text for me" job. It carries the prompt
/// already converted to token ids, how to sample (<see cref="SamplingParameters"/>), and a unique
/// <see cref="RequestId"/> so the engine can interleave your job with many others (continuous batching) and
/// still route each generated token back to the right caller.</para>
/// </remarks>
public sealed class GenerationRequest
{
    /// <summary>Creates a generation request.</summary>
    /// <param name="requestId">A unique id used to correlate outputs back to this request.</param>
    /// <param name="promptTokenIds">The prompt already tokenized to model vocabulary ids.</param>
    /// <param name="samplingParameters">How to decode. Validated on construction.</param>
    /// <param name="arrivalTicks">Monotonic arrival timestamp (e.g. Stopwatch ticks) used by the scheduler
    /// for fairness/priority; pass 0 if the scheduler assigns it.</param>
    public GenerationRequest(
        string requestId,
        IReadOnlyList<int> promptTokenIds,
        SamplingParameters samplingParameters,
        long arrivalTicks = 0)
    {
        if (string.IsNullOrWhiteSpace(requestId))
            throw new ArgumentException("RequestId must be non-empty.", nameof(requestId));
        if (promptTokenIds is null || promptTokenIds.Count == 0)
            throw new ArgumentException("A request must have at least one prompt token.", nameof(promptTokenIds));

        (samplingParameters ?? throw new ArgumentNullException(nameof(samplingParameters))).Validate();

        RequestId = requestId;
        PromptTokenIds = promptTokenIds;
        SamplingParameters = samplingParameters;
        ArrivalTicks = arrivalTicks;
    }

    /// <summary>Unique id correlating this request to its <see cref="RequestOutput"/>s.</summary>
    public string RequestId { get; }

    /// <summary>The tokenized prompt (model vocabulary ids).</summary>
    public IReadOnlyList<int> PromptTokenIds { get; }

    /// <summary>Number of prompt tokens (the prefill length).</summary>
    public int PromptLength => PromptTokenIds.Count;

    /// <summary>How to decode this request.</summary>
    public SamplingParameters SamplingParameters { get; }

    /// <summary>Monotonic arrival timestamp used by the scheduler for ordering/fairness.</summary>
    public long ArrivalTicks { get; }
}
