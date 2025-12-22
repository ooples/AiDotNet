namespace AiDotNet.Serving.Security.ApiKeys;

/// <summary>
/// API key formatting helpers.
/// </summary>
internal static class ApiKeyFormat
{
    private const string Prefix = "aidn";
    private const char Separator = '.';

    public static string Create(string keyId, string secret)
    {
        if (string.IsNullOrWhiteSpace(keyId))
        {
            throw new ArgumentException("KeyId is required.", nameof(keyId));
        }

        if (string.IsNullOrWhiteSpace(secret))
        {
            throw new ArgumentException("Secret is required.", nameof(secret));
        }

        return $"{Prefix}{Separator}{keyId}{Separator}{secret}";
    }

    public static bool TryParse(string apiKey, out string keyId, out string secret)
    {
        keyId = string.Empty;
        secret = string.Empty;

        if (string.IsNullOrWhiteSpace(apiKey))
        {
            return false;
        }

        var trimmed = apiKey.Trim();
        if (!trimmed.StartsWith($"{Prefix}{Separator}", StringComparison.Ordinal))
        {
            return false;
        }

        var first = trimmed.IndexOf(Separator);
        if (first < 0 || first == trimmed.Length - 1)
        {
            return false;
        }

        var second = trimmed.IndexOf(Separator, first + 1);
        if (second < 0 || second == trimmed.Length - 1)
        {
            return false;
        }

        if (trimmed.IndexOf(Separator, second + 1) >= 0)
        {
            return false;
        }

        keyId = trimmed.Substring(first + 1, second - first - 1);
        secret = trimmed.Substring(second + 1);

        if (string.IsNullOrWhiteSpace(keyId) || string.IsNullOrWhiteSpace(secret))
        {
            keyId = string.Empty;
            secret = string.Empty;
            return false;
        }

        return true;
    }
}
