using System.Security.Cryptography;
using System.Text;

namespace AiDotNet.Data.Quality;

/// <summary>
/// Detects exact duplicate documents using cryptographic hashing (SHA-256).
/// </summary>
/// <remarks>
/// <para>
/// The simplest and fastest deduplication method. Uses SHA-256 hashes to find
/// byte-identical documents. Optionally normalizes whitespace and case first.
/// </para>
/// </remarks>
public class ExactHashDeduplicator
{
    private readonly ExactHashDeduplicatorOptions _options;

    public ExactHashDeduplicator(ExactHashDeduplicatorOptions? options = null)
    {
        _options = options ?? new ExactHashDeduplicatorOptions();
    }

    /// <summary>
    /// Computes a hash for the given text.
    /// </summary>
    public string ComputeHash(string text)
    {
        string normalized = text;
        if (_options.CaseInsensitive)
            normalized = normalized.ToLowerInvariant();
        if (_options.NormalizeWhitespace)
            normalized = System.Text.RegularExpressions.Regex.Replace(normalized, @"\s+", " ").Trim();

        using var sha = SHA256.Create();
        byte[] hashBytes = sha.ComputeHash(Encoding.UTF8.GetBytes(normalized));
        return Convert.ToBase64String(hashBytes);
    }

    /// <summary>
    /// Finds duplicate indices from a collection of documents.
    /// </summary>
    /// <param name="documents">The documents to deduplicate.</param>
    /// <returns>Set of indices that are duplicates (should be removed).</returns>
    public HashSet<int> FindDuplicates(IReadOnlyList<string> documents)
    {
        var seen = new Dictionary<string, int>();
        var duplicates = new HashSet<int>();

        for (int i = 0; i < documents.Count; i++)
        {
            string hash = ComputeHash(documents[i]);
            if (seen.ContainsKey(hash))
            {
                duplicates.Add(i);
            }
            else
            {
                seen[hash] = i;
            }
        }

        return duplicates;
    }
}
