using System.Security.Cryptography;

namespace AiDotNet.Serving.Sandboxing;

public static class DockerPasswordGenerator
{
    public static string Generate(int bytes = 24)
    {
        if (bytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(bytes), "Bytes must be > 0.");
        }

        var buffer = RandomNumberGenerator.GetBytes(bytes);
        var base64 = Convert.ToBase64String(buffer);
        return base64.TrimEnd('=').Replace('+', '-').Replace('/', '_');
    }
}

