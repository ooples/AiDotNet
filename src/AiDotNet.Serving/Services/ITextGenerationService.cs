using System.Threading;
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
}
