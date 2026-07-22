using System;

namespace AiDotNet.Serving.Controllers;

/// <summary>
/// Thrown by the OpenAI controller's batch-generation helper when the generation engine reports a failure
/// (via <c>SpeculativeDecodingResponse.Error</c>) that is not a client cancellation. The controller maps it
/// to an HTTP 500 so an engine failure is never masked as a successful (but truncated) 200 response.
/// </summary>
internal sealed class GenerationFailedException : Exception
{
    public GenerationFailedException(string message) : base(message)
    {
    }
}
