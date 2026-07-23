using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models;

namespace AiDotNet.Serving.Services;

/// <summary>
/// Service that performs autoregressive text generation for served language models,
/// driving the continuous-batching engine (with optional speculative decoding).
/// </summary>
/// <remarks>
/// <para>
/// This is the serving-side entry point for token generation. It resolves the requested model
/// (which must support token-level generation), configures the continuous-batching engine from
/// the request, runs the generation loop to completion, and returns the generated tokens.
/// </para>
/// </remarks>
public interface ITextGenerationService
{
    /// <summary>
    /// Generates text by continuing from the request's input tokens.
    /// </summary>
    /// <param name="modelName">The name of the model to generate with.</param>
    /// <param name="numericType">The numeric type the model was loaded with.</param>
    /// <param name="request">The generation request (input tokens, limits, sampling, speculation).</param>
    /// <param name="cancellationToken">
    /// Cancels the synchronous generation loop (e.g. from <c>HttpContext.RequestAborted</c> on client
    /// disconnect) so a long-running request stops consuming engine steps instead of running to budget.
    /// </param>
    /// <returns>
    /// A response containing the generated tokens and statistics. If the model does not support
    /// generation, the response's <see cref="SpeculativeDecodingResponse.Error"/> is populated.
    /// </returns>
    SpeculativeDecodingResponse Generate(string modelName, NumericType numericType, SpeculativeDecodingRequest request, CancellationToken cancellationToken = default);

    /// <summary>
    /// Asynchronous variant of <see cref="Generate"/>. Request-serving controllers must use this so the
    /// ASP.NET request thread is released to the thread pool while the shared batching engine drives the
    /// completion, rather than being blocked for the whole generation (which starves the pool under load).
    /// </summary>
    /// <returns>A task producing the generation response (see <see cref="Generate"/> for the contract).</returns>
    Task<SpeculativeDecodingResponse> GenerateAsync(string modelName, NumericType numericType, SpeculativeDecodingRequest request, CancellationToken cancellationToken = default);

    /// <summary>
    /// Returns whether the named model (loaded with <paramref name="numericType"/>) supports
    /// token-level text generation. Used by the OpenAI-compatible layer to fail fast before it
    /// starts writing a streaming response.
    /// </summary>
    bool SupportsGeneration(string modelName, NumericType numericType);

    /// <summary>
    /// Streams generated token IDs one at a time (true incremental decode with sampling), enabling
    /// real time-to-first-token and Server-Sent-Events responses. Honors
    /// <see cref="SpeculativeDecodingRequest.Temperature"/>, <see cref="SpeculativeDecodingRequest.TopP"/>,
    /// and <see cref="SpeculativeDecodingRequest.TopK"/>. Enumeration is lazy; each <c>MoveNext</c>
    /// advances one decode step. Stops at EOS, at <see cref="SpeculativeDecodingRequest.MaxNewTokens"/>,
    /// or when <paramref name="cancellationToken"/> is signaled.
    /// </summary>
    System.Collections.Generic.IEnumerable<int> GenerateStream(string modelName, NumericType numericType, SpeculativeDecodingRequest request, CancellationToken cancellationToken = default);
}
