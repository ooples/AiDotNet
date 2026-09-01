using System.Security.Cryptography;
using System.Text;

namespace AiDotNet.Evolution;

internal static class EvolutionHash
{
    public static string Compute(string value)
    {
        if (value is null) throw new ArgumentNullException(nameof(value));
        using (SHA256 sha = SHA256.Create())
        {
            byte[] hash = sha.ComputeHash(Encoding.UTF8.GetBytes(value));
            var result = new StringBuilder(hash.Length * 2);
            foreach (byte item in hash) result.Append(item.ToString("x2", System.Globalization.CultureInfo.InvariantCulture));
            return result.ToString();
        }
    }

    public static string Combine(IEnumerable<string> values)
    {
        if (values is null) throw new ArgumentNullException(nameof(values));
        var builder = new StringBuilder();
        foreach (string value in values)
        {
            if (value is null) throw new ArgumentException("Hash components cannot be null.", nameof(values));
            builder.Append(value.Length.ToString(System.Globalization.CultureInfo.InvariantCulture))
                .Append(':').Append(value).Append(';');
        }
        return Compute(builder.ToString());
    }
}
