#if AIDN_LEGACY_TFM
using System;
using System.Net;

namespace AiDotNet.Helpers;

/// <summary>
/// <see cref="WebClient"/> with a bounded request and read/write timeout, for the targets without
/// <c>System.Net.Http.HttpClient</c> (net462, net471, netstandard2.0). A plain WebClient waits up to 100 s per request
/// (and indefinitely while reading), where the modern HttpClient paths it replaces are bounded, so a stalled license
/// server held validation and the background refresh thread far longer than on the modern targets.
/// </summary>
internal sealed class TimedWebClient : WebClient
{
    private readonly int _timeoutMilliseconds;

    internal TimedWebClient(TimeSpan timeout)
    {
        if (timeout <= TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(timeout));
        _timeoutMilliseconds = (int)Math.Min(int.MaxValue, timeout.TotalMilliseconds);
    }

    protected override WebRequest GetWebRequest(Uri address)
    {
        var request = base.GetWebRequest(address);
        request.Timeout = _timeoutMilliseconds;
        if (request is HttpWebRequest http) http.ReadWriteTimeout = _timeoutMilliseconds;
        return request;
    }
}
#endif
