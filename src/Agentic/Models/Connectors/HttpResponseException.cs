using System.Net;
using System.Net.Http;

namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>
/// An <see cref="HttpRequestException"/> that preserves the HTTP status code of a non-success API response on
/// every target framework. On .NET Framework, <see cref="HttpRequestException"/> exposes no status code, so
/// connectors throw this instead — keeping the status available to retry classification
/// (<c>ChatClientBase.IsRetryable</c>) without it being lost into a plain message string.
/// </summary>
/// <remarks>
/// The property is named <see cref="ResponseStatusCode"/> (not <c>StatusCode</c>) to avoid hiding the
/// nullable <c>HttpRequestException.StatusCode</c> that exists on modern frameworks.
/// </remarks>
public sealed class HttpResponseException : HttpRequestException
{
    /// <summary>
    /// Initializes a new instance carrying the failing response's status code.
    /// </summary>
    /// <param name="statusCode">The HTTP status code of the non-success response.</param>
    /// <param name="message">A descriptive error message.</param>
    public HttpResponseException(HttpStatusCode statusCode, string message)
        : base(message)
    {
        ResponseStatusCode = statusCode;
    }

    /// <summary>Gets the HTTP status code of the non-success response.</summary>
    public HttpStatusCode ResponseStatusCode { get; }
}
