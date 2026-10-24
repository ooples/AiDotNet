using System;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace AiDotNet.Helpers;

/// <summary>
/// The one <see cref="HttpClient"/> the library's one-off downloads share: dataset archives, pretrained weights and
/// ONNX models.
/// </summary>
/// <remarks>
/// <para>
/// Each download used to create and dispose a client of its own. A disposed client leaves its sockets in
/// <c>TIME_WAIT</c>, so repeated downloads exhaust ephemeral ports, and every request paid a fresh TLS handshake.
/// One long-lived client pools its connections instead (#2363).
/// </para>
/// <para>
/// The client has no timeout of its own, because the downloads need different ones (a weight file a couple of
/// minutes, a dataset archive up to an hour). Each call bounds itself with <see cref="WithTimeout"/>.
/// </para>
/// </remarks>
internal static class SharedHttp
{
    /// <summary>The timeout <see cref="HttpClient"/> applies when none is set: 100 seconds.</summary>
    internal static readonly TimeSpan DefaultTimeout = TimeSpan.FromSeconds(100);

    /// <summary>The shared download client. It follows redirects, up to ten.</summary>
    internal static HttpClient Downloads { get; } = Create();

    private static HttpClient Create()
    {
        var handler = new HttpClientHandler
        {
            AllowAutoRedirect = true,
            MaxAutomaticRedirections = 10,
        };
        return new HttpClient(handler) { Timeout = Timeout.InfiniteTimeSpan };
    }

    /// <summary>
    /// A cancellation source that cancels after <paramref name="timeout"/> or when <paramref name="cancellationToken"/>
    /// does. Dispose it when the download ends.
    /// </summary>
    internal static CancellationTokenSource WithTimeout(TimeSpan timeout, CancellationToken cancellationToken)
    {
        var source = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        source.CancelAfter(timeout);
        return source;
    }

    /// <summary>
    /// Copies <paramref name="response"/>'s body to <paramref name="destination"/>, stopping when
    /// <paramref name="cancellationToken"/> is cancelled, on every target framework.
    /// </summary>
    /// <remarks>
    /// With <see cref="HttpCompletionOption.ResponseHeadersRead"/> the request completes at the headers, so a timeout
    /// that bounds only the request leaves a stalled body unbounded. <c>HttpContent.CopyToAsync</c> takes a token only
    /// from .NET 5, and a network read on .NET Framework may not observe one, so cancellation also disposes the
    /// response, which aborts a read in progress.
    /// </remarks>
    internal static async Task CopyBodyAsync(HttpResponseMessage response, Stream destination, CancellationToken cancellationToken)
    {
        if (response is null) throw new ArgumentNullException(nameof(response));
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        using (cancellationToken.Register(response.Dispose))
        using (var body = await response.Content.ReadAsStreamAsync().ConfigureAwait(false))
        {
            await body.CopyToAsync(destination, 81920, cancellationToken).ConfigureAwait(false);
        }
        cancellationToken.ThrowIfCancellationRequested();
    }
}
