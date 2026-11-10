using System.Diagnostics.CodeAnalysis;
using System.Security.Cryptography;
using System.Text;
using AiDotNet.Serving.Configuration;

namespace AiDotNet.Serving.Security.ApiKeys;

/// <summary>
/// Creates and verifies config-backed (static) API keys, configured under
/// <c>ServingSecurity:ApiKeys:StaticApiKeys</c>.
/// </summary>
/// <remarks>
/// A static key has the same <c>aidn.&lt;keyId&gt;.&lt;secret&gt;</c> form as a database key. Its configuration
/// entry holds the key id, a PBKDF2-SHA256 hash of the decoded secret and, when HMAC secrets are configured, the
/// HMAC-SHA256 of the key id. Use <see cref="Generate"/> to mint a key and the entry that accepts it.
/// </remarks>
public static class StaticApiKeys
{
    private const int KeyIdBytes = 8;
    private const int SecretBytes = 32;
    private const int SaltBytes = 16;
    private const int HashBytes = 32;

    /// <summary>
    /// Creates a new API key and the configuration entry that accepts it.
    /// </summary>
    /// <param name="tier">The tier the key grants.</param>
    /// <param name="pbkdf2Iterations">The PBKDF2 iteration count to hash the secret with.</param>
    /// <param name="hmacSecret">The newest of <see cref="ServingApiKeyOptions.HmacSecrets"/>, or
    /// <see langword="null"/> when none are configured.</param>
    /// <param name="expiresAt">When the key expires, or <see langword="null"/> for no expiry.</param>
    /// <returns>The API key to hand to the client (shown once; it is not recoverable from the entry) and the
    /// entry to add to the configuration.</returns>
    public static (string ApiKey, StaticApiKeyOptions Entry) Generate(
        ServingTier tier,
        int pbkdf2Iterations,
        string? hmacSecret = null,
        DateTimeOffset? expiresAt = null)
    {
        if (pbkdf2Iterations < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(pbkdf2Iterations), "The iteration count must be positive.");
        }

        string keyId = Base64Url.Encode(RandomNumberGenerator.GetBytes(KeyIdBytes));
        byte[] secret = RandomNumberGenerator.GetBytes(SecretBytes);
        byte[] salt = RandomNumberGenerator.GetBytes(SaltBytes);
        try
        {
            var entry = new StaticApiKeyOptions
            {
                Prefix = keyId,
                HmacIdBase64 = string.IsNullOrWhiteSpace(hmacSecret) ? string.Empty : ComputeHmacIdBase64(hmacSecret, keyId),
                Pbkdf2SaltBase64 = Convert.ToBase64String(salt),
                Pbkdf2HashBase64 = Convert.ToBase64String(Hash(secret, salt, pbkdf2Iterations)),
                Pbkdf2Iterations = pbkdf2Iterations,
                Tier = tier,
                ExpiresAt = expiresAt
            };
            return (ApiKeyFormat.Create(keyId, Base64Url.Encode(secret)), entry);
        }
        finally
        {
            CryptographicOperations.ZeroMemory(secret);
        }
    }

    /// <summary>
    /// Computes the base64 HMAC-SHA256 of <paramref name="keyId"/> under <paramref name="hmacSecret"/>, both UTF-8.
    /// </summary>
    /// <param name="hmacSecret">One of <see cref="ServingApiKeyOptions.HmacSecrets"/>.</param>
    /// <param name="keyId">The key id (<see cref="StaticApiKeyOptions.Prefix"/>).</param>
    /// <returns>The value for <see cref="StaticApiKeyOptions.HmacIdBase64"/>.</returns>
    public static string ComputeHmacIdBase64(string hmacSecret, string keyId)
    {
        if (string.IsNullOrWhiteSpace(hmacSecret))
        {
            throw new ArgumentException("The HMAC secret is required.", nameof(hmacSecret));
        }

        if (string.IsNullOrWhiteSpace(keyId))
        {
            throw new ArgumentException("The key id is required.", nameof(keyId));
        }

        return Convert.ToBase64String(HMACSHA256.HashData(Encoding.UTF8.GetBytes(hmacSecret), Encoding.UTF8.GetBytes(keyId)));
    }

    /// <summary>
    /// Looks <paramref name="keyId"/> up among the configured static keys and verifies the presented secret.
    /// </summary>
    /// <param name="options">The bound API key options.</param>
    /// <param name="keyId">The key id parsed from the presented key.</param>
    /// <param name="secret">The decoded secret parsed from the presented key.</param>
    /// <param name="now">The current time, for the expiry check.</param>
    /// <param name="owned"><see langword="true"/> when a static entry owns <paramref name="keyId"/>, whether or not
    /// it accepted the key; <see langword="false"/> when no static entry has that id.</param>
    /// <param name="result">The authenticated key when the method returns <see langword="true"/>.</param>
    /// <returns><see langword="true"/> when a static entry owns <paramref name="keyId"/> and accepted the key.</returns>
    internal static bool TryAuthenticate(
        ServingApiKeyOptions options,
        string keyId,
        byte[] secret,
        DateTimeOffset now,
        out bool owned,
        [NotNullWhen(true)] out ApiKeyAuthenticationResult? result)
    {
        result = null;
        StaticApiKeyOptions? entry = options.StaticApiKeys.Find(e => string.Equals(e.Prefix, keyId, StringComparison.Ordinal));
        owned = entry is not null;
        if (entry is null)
        {
            return false;
        }

        if (entry.Revoked || (entry.ExpiresAt is DateTimeOffset expiresAt && expiresAt <= now))
        {
            return false;
        }

        if (options.HmacSecrets.Count > 0 && !HmacIdMatches(options.HmacSecrets, entry))
        {
            return false;
        }

        byte[] salt = Convert.FromBase64String(entry.Pbkdf2SaltBase64);
        byte[] expected = Convert.FromBase64String(entry.Pbkdf2HashBase64);
        byte[] computed = Hash(secret, salt, entry.Pbkdf2Iterations, expected.Length);
        bool ok = CryptographicOperations.FixedTimeEquals(expected, computed);
        CryptographicOperations.ZeroMemory(computed);
        if (!ok)
        {
            return false;
        }

        SubscriptionTier tier = ToSubscriptionTier(entry.Tier);
        result = new ApiKeyAuthenticationResult(entry.Prefix, tier, ApiKeyService.DefaultScopesForTier(tier));
        return true;
    }

    /// <summary>
    /// Maps the configuration tier of a static key to the subscription tier its callers get.
    /// </summary>
    internal static SubscriptionTier ToSubscriptionTier(ServingTier tier) => tier switch
    {
        ServingTier.Free => SubscriptionTier.Free,
        ServingTier.Premium => SubscriptionTier.Pro,
        ServingTier.Enterprise => SubscriptionTier.Enterprise,
        _ => throw new ArgumentOutOfRangeException(nameof(tier), tier, "Unknown serving tier.")
    };

    private static bool HmacIdMatches(List<string> hmacSecrets, StaticApiKeyOptions entry)
    {
        byte[] expected = Convert.FromBase64String(entry.HmacIdBase64);
        bool match = false;
        foreach (string hmacSecret in hmacSecrets)
        {
            byte[] computed = Convert.FromBase64String(ComputeHmacIdBase64(hmacSecret, entry.Prefix));
            // Every secret is tried, so the time taken does not reveal which one matched.
            match |= CryptographicOperations.FixedTimeEquals(expected, computed);
        }

        return match;
    }

    private static byte[] Hash(byte[] secret, byte[] salt, int iterations, int bytes = HashBytes) =>
        Rfc2898DeriveBytes.Pbkdf2(secret, salt, iterations, HashAlgorithmName.SHA256, bytes);
}
