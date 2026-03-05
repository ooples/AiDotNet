using System.Security.Cryptography;

namespace AiDotNet.Helpers;

/// <summary>
/// Runtime integrity checker that verifies the assembly has not been tampered with.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> During official CI/CD builds, a hash of the DLL is computed and signed
/// with the build key, then embedded as a resource. At runtime, before any crypto operation,
/// this checker verifies that the embedded build key resource is valid and that critical types
/// (like <see cref="ModelPayloadEncryption"/>) still exist in the assembly.
///
/// If the assembly has been tampered with (e.g., someone replaced the build key or removed
/// the encryption module), this check fails and crypto operations are blocked.
///
/// Dev/fork builds without an embedded integrity hash always pass (to allow development).
/// This is Layer 3 of the three-layer obfuscation system.
/// </remarks>
internal static class AssemblyIntegrityChecker
{
    private const string IntegrityResourceName = "AiDotNet.IntegrityHash";
    private static bool? _cachedResult;
    private static readonly object _lock = new();

    /// <summary>
    /// Verifies the runtime integrity of the assembly.
    /// </summary>
    /// <returns>
    /// True if integrity is verified or if this is a dev build (no integrity hash embedded).
    /// False only if the integrity hash is present but verification fails.
    /// </returns>
    internal static bool VerifyIntegrity()
    {
        if (_cachedResult.HasValue)
        {
            return _cachedResult.Value;
        }

        lock (_lock)
        {
            if (_cachedResult.HasValue)
            {
                return _cachedResult.Value;
            }

            _cachedResult = VerifyIntegrityCore();
            return _cachedResult.Value;
        }
    }

    private static bool VerifyIntegrityCore()
    {
        // If no build key is present, this is a dev/fork build — allow
        var buildKey = BuildKeyProvider.GetBuildKey();
        if (buildKey.Length == 0)
        {
            return true;
        }

        // Check that critical types still exist in the assembly
        // (prevents someone from gutting the encryption module)
        var assembly = typeof(AssemblyIntegrityChecker).Assembly;
        var encryptionType = assembly.GetType("AiDotNet.Helpers.ModelPayloadEncryption");
        if (encryptionType is null)
        {
            return false;
        }

        var buildKeyProviderType = assembly.GetType("AiDotNet.Helpers.BuildKeyProvider");
        if (buildKeyProviderType is null)
        {
            return false;
        }

        // Check for embedded integrity hash
        using var stream = assembly.GetManifestResourceStream(IntegrityResourceName);
        if (stream is null || stream.Length == 0)
        {
            // No integrity hash embedded — this is valid for builds where integrity
            // hashing is not yet configured. Allow if build key is present.
            return true;
        }

        // Read the expected HMAC from the embedded resource
        if (stream.Length > int.MaxValue)
        {
            return false;
        }

        var expectedHmac = new byte[checked((int)stream.Length)];
        int bytesRead = 0;
        while (bytesRead < expectedHmac.Length)
        {
            int read = stream.Read(expectedHmac, bytesRead, expectedHmac.Length - bytesRead);
            if (read == 0)
            {
                break;
            }

            bytesRead += read;
        }

        // Self-consistency check: verify the embedded resource was produced from this build key.
        // The CI/CD pipeline computes HMAC-SHA256(buildKey, buildKey) and embeds the result.
        // At runtime we recompute the same HMAC and compare. This confirms the build key
        // resource has not been swapped without also replacing the key itself.
        // NOTE: This does NOT verify the DLL contents at runtime — that would require
        // hashing the entire assembly on every load. Full DLL integrity is verified at
        // build time by CI/CD.
        try
        {
            using var hmac = new HMACSHA256(buildKey);
            var selfCheck = hmac.ComputeHash(buildKey);

            // Verify the first 32 bytes match (HMAC-SHA256 output)
            if (expectedHmac.Length < 32 || selfCheck.Length < 32)
            {
                return false;
            }

#if NET471
            // Constant-time comparison for net471
            bool match = true;
            for (int i = 0; i < 32; i++)
            {
                match &= expectedHmac[i] == selfCheck[i];
            }

            return match;
#else
            return CryptographicOperations.FixedTimeEquals(
                expectedHmac.AsSpan(0, 32),
                selfCheck.AsSpan(0, 32));
#endif
        }
        catch
        {
            return false;
        }
    }
}
