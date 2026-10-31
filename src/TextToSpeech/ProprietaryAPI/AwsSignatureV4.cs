using System.Net.Http;
using System.Security.Cryptography;
using System.Text;

namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>
/// AWS Signature Version 4 request signing (AWS General Reference, "Signing AWS API requests"): the canonical request,
/// the string to sign over its SHA-256 hash, the key derived from the secret by HMAC-SHA256 over the date, region,
/// service and "aws4_request", and the Authorization header.
/// </summary>
internal static class AwsSignatureV4
{
    /// <summary>Signs <paramref name="request"/> in place: adds <c>x-amz-date</c>, <c>x-amz-content-sha256</c>, the
    /// session token when given, and <c>Authorization</c>. Every header present on the request is signed.</summary>
    public static void Sign(HttpRequestMessage request, byte[] body, string service, string region, string accessKeyId,
        string secretAccessKey, string? sessionToken, DateTime utcNow, bool includeContentHash = true)
    {
        string amzDate = utcNow.ToString("yyyyMMdd'T'HHmmss'Z'", System.Globalization.CultureInfo.InvariantCulture);
        string date = amzDate.Substring(0, 8);
        string payloadHash = Hex(Sha256(body));
        request.Headers.Remove("x-amz-date");
        request.Headers.TryAddWithoutValidation("x-amz-date", amzDate);
        if (includeContentHash)
        {
            request.Headers.Remove("x-amz-content-sha256");
            request.Headers.TryAddWithoutValidation("x-amz-content-sha256", payloadHash);
        }
        if (!string.IsNullOrEmpty(sessionToken))
        {
            request.Headers.Remove("x-amz-security-token");
            request.Headers.TryAddWithoutValidation("x-amz-security-token", sessionToken);
        }

        var uri = request.RequestUri ?? throw new InvalidOperationException("The request has no URI.");
        var headers = new SortedDictionary<string, string>(StringComparer.Ordinal)
        {
            ["host"] = uri.IsDefaultPort ? uri.Host : $"{uri.Host}:{uri.Port}",
        };
        foreach (var header in request.Headers)
            headers[header.Key.ToLowerInvariant()] = string.Join(",", header.Value.Select(Trim));
        if (request.Content is not null)
            foreach (var header in request.Content.Headers)
                headers[header.Key.ToLowerInvariant()] = string.Join(",", header.Value.Select(Trim));
        headers.Remove("authorization");

        string signedHeaders = string.Join(";", headers.Keys);
        string canonicalRequest = string.Join("\n",
            request.Method.Method,
            CanonicalPath(uri),
            CanonicalQuery(uri),
            string.Concat(headers.Select(h => $"{h.Key}:{h.Value}\n")),
            signedHeaders,
            payloadHash);
        string scope = $"{date}/{region}/{service}/aws4_request";
        string stringToSign = $"AWS4-HMAC-SHA256\n{amzDate}\n{scope}\n{Hex(Sha256(Encoding.UTF8.GetBytes(canonicalRequest)))}";

        byte[] key = Hmac(Encoding.UTF8.GetBytes("AWS4" + secretAccessKey), date);
        key = Hmac(key, region);
        key = Hmac(key, service);
        key = Hmac(key, "aws4_request");
        string signature = Hex(Hmac(key, stringToSign));
        request.Headers.TryAddWithoutValidation("Authorization",
            $"AWS4-HMAC-SHA256 Credential={accessKeyId}/{scope}, SignedHeaders={signedHeaders}, Signature={signature}");
    }

    // Header values are trimmed and their inner runs of spaces collapsed.
    private static string Trim(string value) => System.Text.RegularExpressions.Regex.Replace(value.Trim(), " +", " ");

    // Each path segment URI-encoded (RFC 3986 unreserved characters kept); "/" for an empty path.
    private static string CanonicalPath(Uri uri)
    {
        string path = uri.AbsolutePath;
        if (string.IsNullOrEmpty(path)) return "/";
        return string.Join("/", path.Split('/').Select(segment => Encode(Uri.UnescapeDataString(segment))));
    }

    // Parameters sorted by name then value, each name and value URI-encoded.
    private static string CanonicalQuery(Uri uri)
    {
        string query = uri.Query.TrimStart('?');
        if (query.Length == 0) return string.Empty;
        var pairs = query.Split('&').Where(p => p.Length > 0).Select(p =>
        {
            int eq = p.IndexOf('=');
            string name = Uri.UnescapeDataString(eq < 0 ? p : p.Substring(0, eq));
            string value = eq < 0 ? string.Empty : Uri.UnescapeDataString(p.Substring(eq + 1));
            return (Name: Encode(name), Value: Encode(value));
        });
        return string.Join("&", pairs.OrderBy(p => p.Name, StringComparer.Ordinal).ThenBy(p => p.Value, StringComparer.Ordinal)
            .Select(p => $"{p.Name}={p.Value}"));
    }

    private static string Encode(string value)
    {
        var sb = new StringBuilder();
        foreach (byte b in Encoding.UTF8.GetBytes(value))
        {
            char c = (char)b;
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~')
                sb.Append(c);
            else
                sb.Append('%').Append(b.ToString("X2", System.Globalization.CultureInfo.InvariantCulture));
        }
        return sb.ToString();
    }

    private static byte[] Sha256(byte[] data)
    {
        using var sha = SHA256.Create();
        return sha.ComputeHash(data);
    }

    private static byte[] Hmac(byte[] key, string data)
    {
        using var hmac = new HMACSHA256(key);
        return hmac.ComputeHash(Encoding.UTF8.GetBytes(data));
    }

    internal static string Hex(byte[] bytes)
    {
        var sb = new StringBuilder(bytes.Length * 2);
        foreach (byte b in bytes) sb.Append(b.ToString("x2", System.Globalization.CultureInfo.InvariantCulture));
        return sb.ToString();
    }
}
