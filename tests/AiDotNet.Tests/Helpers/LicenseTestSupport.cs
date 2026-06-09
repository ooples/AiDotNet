using System;
using System.Security.Cryptography;
using System.Text;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.Helpers;

/// <summary>
/// Test support for the fail-closed offline license path. After the security fix, offline validation
/// grants a license ONLY when a key's HMAC-SHA256 signature verifies against the embedded build key, so
/// tests can no longer use hardcoded well-formed-but-unsigned strings (those now correctly fail). This
/// helper injects a known test build key and produces REAL signed keys against it, so tests exercise the
/// genuine cryptographic path. Scope the injected key with <see cref="WithBuildKey"/> inside a collection
/// that disables parallelization (the build key is process-global).
/// </summary>
internal static class LicenseTestSupport
{
    /// <summary>A fixed 40-byte test build key (never a real signing secret).</summary>
    internal static readonly byte[] TestBuildKey =
        Encoding.UTF8.GetBytes("aidotnet-test-build-key-0123456789ABCDEF");

    /// <summary>
    /// Produces a valid signed key <c>aidn.{id}.{sig}</c> where
    /// <c>sig = base64url(HMACSHA256(buildKey, "aidn.{id}"))</c> — i.e. exactly what ValidateOffline()
    /// recomputes and compares. The id must be base64url-safe (letters/digits/-/_).
    /// </summary>
    internal static string SignedKey(string id, byte[]? buildKey = null)
    {
        string payload = "aidn." + id;
        using var hmac = new HMACSHA256(buildKey ?? TestBuildKey);
        byte[] sig = hmac.ComputeHash(Encoding.UTF8.GetBytes(payload));
        string b64url = Convert.ToBase64String(sig).Replace('+', '-').Replace('/', '_').TrimEnd('=');
        return payload + "." + b64url;
    }

    /// <summary>
    /// Snapshots the build key currently in effect (the collection fixture's injected key, or an official
    /// build's embedded key), or null when none is set. Used to scope overrides so they restore the prior
    /// state rather than unconditionally clearing it.
    /// </summary>
    internal static byte[]? CurrentBuildKeySnapshot()
    {
        var key = BuildKeyProvider.GetBuildKey();
        return key.Length > 0 ? key : null;
    }

    /// <summary>
    /// Scopes the embedded build key for the duration of a test, restoring the PREVIOUS key on dispose
    /// (not a hard clear). Pass null to simulate a dev/fork build (no build key) and assert fail-closed
    /// behaviour — the prior key is still restored afterwards, so this is safe inside the License
    /// collection fixture and won't strip the injected key from later tests.
    /// </summary>
    internal static IDisposable WithBuildKey(byte[]? key = null)
    {
        var previous = CurrentBuildKeySnapshot();
        BuildKeyProvider.OverrideForTesting(key ?? TestBuildKey);
        return new Restore(previous);
    }

    private sealed class Restore : IDisposable
    {
        private readonly byte[]? _previous;

        public Restore(byte[]? previous) =>
            _previous = previous is { Length: > 0 } ? (byte[])previous.Clone() : null;

        public void Dispose() => BuildKeyProvider.OverrideForTesting(_previous);
    }
}

/// <summary>
/// Injects the test build key for the lifetime of the License test collection so offline HMAC validation
/// can succeed, and clears it afterwards. The build key is process-global, so the collection disables
/// parallelization (see <see cref="LicenseCollection"/>) to avoid leaking into unrelated tests that assert
/// dev/fork (no-build-key) behaviour.
/// </summary>
public sealed class LicenseBuildKeyFixture : IDisposable
{
    // Snapshot whatever key was in effect before the fixture (an official build's embedded key, or none)
    // so teardown restores it rather than clearing — an official-build run keeps its embedded key afterwards.
    private readonly byte[]? _previousKey = LicenseTestSupport.CurrentBuildKeySnapshot();

    public LicenseBuildKeyFixture() => BuildKeyProvider.OverrideForTesting(LicenseTestSupport.TestBuildKey);

    public void Dispose() => BuildKeyProvider.OverrideForTesting(_previousKey);
}

[CollectionDefinition("License", DisableParallelization = true)]
public sealed class LicenseCollection : Xunit.ICollectionFixture<LicenseBuildKeyFixture>
{
}
