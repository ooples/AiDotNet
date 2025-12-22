namespace AiDotNet.Serving.Security.ApiKeys;

internal static class Base64Url
{
    public static string Encode(byte[] data)
    {
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        return Convert.ToBase64String(data)
            .TrimEnd('=')
            .Replace('+', '-')
            .Replace('/', '_');
    }

    public static bool TryDecode(string text, out byte[] data)
    {
        data = Array.Empty<byte>();
        if (string.IsNullOrWhiteSpace(text))
        {
            return false;
        }

        string s = text.Trim()
            .Replace('-', '+')
            .Replace('_', '/');

        switch (s.Length % 4)
        {
            case 0:
                break;
            case 2:
                s += "==";
                break;
            case 3:
                s += "=";
                break;
            default:
                return false;
        }

        try
        {
            data = Convert.FromBase64String(s);
            return true;
        }
        catch
        {
            return false;
        }
    }
}

