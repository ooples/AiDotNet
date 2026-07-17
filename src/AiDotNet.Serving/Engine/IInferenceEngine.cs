namespace AiDotNet.Serving.Engine;

/// <summary>
/// The core high-throughput inference engine contract: a continuously-batching, KV-cache-managed generation
/// loop. Callers <see cref="AddRequest"/> jobs at any time; the engine interleaves them (continuous / in-flight
/// batching), manages paged KV-cache memory with preemption, and streams <see cref="RequestOutput"/>s back.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> this is the "brain" that actually runs the model for serving. Unlike a simple
/// "call model, get text" function, a serving engine keeps many requests in flight at once and advances all of
/// them a little each <see cref="Step"/>, so the GPU/CPU stays busy and latency stays low. You add requests
/// whenever they arrive; you pump <see cref="Step"/> in a loop; each step returns the newly-produced tokens
/// for every request that advanced.</para>
/// <para>This mirrors vLLM's <c>LLMEngine</c> / TGI's batching server: the design goal is to meet or exceed
/// their throughput and latency by combining paged KV memory, in-flight batching, chunked prefill, and
/// (optionally) speculative decoding and quantized execution behind this single contract.</para>
/// </remarks>
public interface IInferenceEngine : IDisposable
{
    /// <summary>Admits a new request into the engine's waiting queue. Thread-safe; may be called while stepping.</summary>
    void AddRequest(GenerationRequest request);

    /// <summary>
    /// Requests cancellation of an in-flight request by id. The request's sequences move to a terminal
    /// aborted state and their KV blocks are freed at the next step. Returns true if the request was found.
    /// </summary>
    bool AbortRequest(string requestId);

    /// <summary>
    /// Advances the engine by one iteration: schedules a batch (prefill and/or decode), runs the model,
    /// samples one token per running sequence, frees finished sequences, and returns the outputs that changed
    /// this step. Returns an empty list when there is no work.
    /// </summary>
    IReadOnlyList<RequestOutput> Step();

    /// <summary>True while any request is queued or running (i.e. <see cref="Step"/> still has work to do).</summary>
    bool HasUnfinishedRequests { get; }

    /// <summary>A snapshot of current load and KV-cache utilization.</summary>
    EngineStatistics GetStatistics();
}
