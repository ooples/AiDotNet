using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace AiDotNet.Extensions;

/// <summary>
/// Extension methods for <see cref="HttpClient"/> providing cross-platform compatibility.
/// </summary>
public static class HttpClientExtensions
{
    /// <summary>
    /// Downloads the content at the specified URL as a byte array with cancellation support.
    /// </summary>
    /// <param name="client">The HTTP client.</param>
    /// <param name="requestUri">The URL to download from.</param>
    /// <param name="cancellationToken">The cancellation token.</param>
    /// <returns>The downloaded bytes.</returns>
    /// <remarks>
    /// This extension provides a unified API for downloading byte arrays with cancellation
    /// support across all .NET target frameworks. In .NET 5+, the native overload is used.
    /// In older frameworks, GetAsync with cancellation is used followed by ReadAsByteArrayAsync.
    /// </remarks>
    public static async Task<byte[]> GetByteArrayWithCancellationAsync(
        this HttpClient client,
        string requestUri,
        CancellationToken cancellationToken = default)
    {
#if NET5_0_OR_GREATER
        return await client.GetByteArrayAsync(requestUri, cancellationToken).ConfigureAwait(false);
#else
        using var response = await client.GetAsync(requestUri, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadAsByteArrayAsync().ConfigureAwait(false);
#endif
    }
}
