using System;
using System.Collections.Generic;
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
    /// <summary>
    /// A fixed 32-byte test build key (never a real signing secret). The size matches the
    /// official build key (HMAC-SHA256 key length) so tests asserting the published
    /// "0 bytes (dev) or 32 bytes (official)" key-size contract pass under the
    /// ModuleInitializer's process-wide override.
    /// </summary>
    internal static readonly byte[] TestBuildKey =
        Encoding.UTF8.GetBytes("aidotnet-test-build-key-32bytes!");

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

    // ───────────────────────── Asymmetric (aidn2) test support ─────────────────────────

    /// <summary>
    /// A fixed key-id for the ephemeral test signing keypair. Real builds embed CI-generated kids.
    /// </summary>
    internal const string TestKid = "test-kid-2026a";

    /// <summary>
    /// Ephemeral Ed25519 test keypair (NEVER a real signing key). Its PUBLIC half (32 bytes) is injected
    /// into <see cref="LicensePublicKeyProvider"/> by the License collection fixture; its PRIVATE half
    /// signs test tokens via <see cref="SignedKeyV2"/>. Regenerated per test process — nothing committed.
    /// </summary>
    private static readonly Org.BouncyCastle.Crypto.AsymmetricCipherKeyPair TestKeyPair = CreateEd25519KeyPair();

    private static Org.BouncyCastle.Crypto.AsymmetricCipherKeyPair CreateEd25519KeyPair()
    {
        var generator = new Org.BouncyCastle.Crypto.Generators.Ed25519KeyPairGenerator();
        generator.Init(new Org.BouncyCastle.Crypto.Parameters.Ed25519KeyGenerationParameters(
            new Org.BouncyCastle.Security.SecureRandom()));
        return generator.GenerateKeyPair();
    }

    /// <summary>The raw 32-byte Ed25519 PUBLIC key of the ephemeral test signing keypair.</summary>
    internal static byte[] TestPublicKey =>
        ((Org.BouncyCastle.Crypto.Parameters.Ed25519PublicKeyParameters)TestKeyPair.Public).GetEncoded();

    /// <summary>The default embedded public-key set used by the License fixture: { TestKid → test public key }.</summary>
    internal static Dictionary<string, byte[]> DefaultTestKeySet() =>
        new(StringComparer.Ordinal) { [TestKid] = TestPublicKey };

    /// <summary>
    /// Produces a valid asymmetric token <c>aidn2.{base64url(claims)}.{base64url(sig)}</c> signed with the
    /// ephemeral test PRIVATE key (Ed25519 / EdDSA), exactly what <see cref="AsymmetricLicenseVerifier"/>
    /// verifies. Pass a foreign <paramref name="signingKey"/> to forge a bad-signature token, or an
    /// <paramref name="exp"/> in the past for an expired token.
    /// </summary>
    internal static string SignedKeyV2(
        string sub = "test-customer",
        string tier = "pro",
        int seats = 5,
        DateTimeOffset? iat = null,
        DateTimeOffset? exp = null,
        string? kid = null,
        string? alg = "EdDSA",
        Org.BouncyCastle.Crypto.AsymmetricKeyParameter? signingKey = null,
        string? jti = null,
        string[]? caps = null,
        string? mach = null,
        string? scope = null)
    {
        var claims = new LicenseClaims
        {
            Sub = sub,
            Tier = tier,
            Seats = seats,
            Iat = (iat ?? DateTimeOffset.UtcNow).ToUnixTimeSeconds(),
            Exp = (exp ?? DateTimeOffset.UtcNow.AddDays(30)).ToUnixTimeSeconds(),
            Kid = kid ?? TestKid,
            Alg = alg,
            Jti = jti,
            Caps = caps,
            Mach = mach,
            Scope = scope
        };

        byte[] claimsBytes = Encoding.UTF8.GetBytes(claims.ToCanonicalJson());
        var signer = new Org.BouncyCastle.Crypto.Signers.Ed25519Signer();
        signer.Init(true, signingKey ?? TestKeyPair.Private);
        signer.BlockUpdate(claimsBytes, 0, claimsBytes.Length);
        byte[] sig = signer.GenerateSignature();

        return AsymmetricLicenseVerifier.Prefix + "." +
               Base64UrlHelper.Encode(claimsBytes) + "." +
               Base64UrlHelper.Encode(sig);
    }

    /// <summary>
    /// Produces a signed revocation list (CRL) in the format <see cref="AiDotNet.Helpers.LicenseRevocationProvider"/>
    /// verifies: <c>{ kid, payload:base64url({iat,exp,rkids,rjti}), sig:base64url(Ed25519 over payload) }</c>,
    /// signed with the ephemeral test PRIVATE key (so it verifies against the injected test public key).
    /// </summary>
    internal static string SignedCrlV2(
        string[]? revokedJti = null,
        string[]? revokedKids = null,
        DateTimeOffset? exp = null,
        string? kid = null,
        Org.BouncyCastle.Crypto.AsymmetricKeyParameter? signingKey = null,
        long? iat = null)
    {
        var payload = new
        {
            // iat is overridable so a test can issue two CRLs with the SAME second (to exercise the
            // same-iat merge path); otherwise it defaults to now.
            iat = iat ?? DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            exp = (exp ?? DateTimeOffset.UtcNow.AddDays(1)).ToUnixTimeSeconds(),
            rkids = revokedKids ?? System.Array.Empty<string>(),
            rjti = revokedJti ?? System.Array.Empty<string>()
        };
        byte[] payloadBytes = Encoding.UTF8.GetBytes(Newtonsoft.Json.JsonConvert.SerializeObject(payload));
        var signer = new Org.BouncyCastle.Crypto.Signers.Ed25519Signer();
        signer.Init(true, signingKey ?? TestKeyPair.Private);
        signer.BlockUpdate(payloadBytes, 0, payloadBytes.Length);
        byte[] sig = signer.GenerateSignature();

        var envelope = new
        {
            kid = kid ?? TestKid,
            payload = Base64UrlHelper.Encode(payloadBytes),
            sig = Base64UrlHelper.Encode(sig)
        };
        return Newtonsoft.Json.JsonConvert.SerializeObject(envelope);
    }

    /// <summary>
    /// Creates a fresh, unrelated Ed25519 PRIVATE key — used to sign a token whose signature will NOT
    /// verify against the embedded test public key (forged-token regression).
    /// </summary>
    internal static Org.BouncyCastle.Crypto.AsymmetricKeyParameter CreateForeignSigningKey() =>
        CreateEd25519KeyPair().Private;

    /// <summary>
    /// Scopes the embedded license public-key set to <paramref name="keys"/> for the duration of a test,
    /// restoring the previous set on dispose. Pass <see langword="null"/> to simulate a dev/fork build with
    /// NO embedded public key and assert fail-closed behaviour.
    /// </summary>
    internal static IDisposable WithLicensePublicKeys(IReadOnlyDictionary<string, byte[]>? keys)
    {
        var previous = LicensePublicKeyProvider.CurrentSnapshot();
        LicensePublicKeyProvider.OverrideForTesting(keys);
        return new RestorePublicKeys(previous);
    }

    private sealed class RestorePublicKeys : IDisposable
    {
        private readonly IReadOnlyDictionary<string, byte[]>? _previous;
        public RestorePublicKeys(IReadOnlyDictionary<string, byte[]>? previous) => _previous = previous;
        public void Dispose() => LicensePublicKeyProvider.OverrideForTesting(_previous);
    }

    /// <summary>
    /// Snapshots the build key currently in effect (the collection fixture's injected key, or an official
    /// build's embedded key), or null when none is set. Used to scope overrides so they restore the prior
    /// state rather than unconditionally clearing it.
    /// </summary>
    internal static byte[]? CurrentBuildKeySnapshot()
    {
        var key = BuildKeyProvider.GetBuildKey();
        // Return an owned copy so callers hold a stable snapshot regardless of how the
        // provider manages its internal buffer (keeps Restore/fixture defensively consistent).
        return key.Length > 0 ? (byte[])key.Clone() : null;
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

    // Same restore discipline for the asymmetric (aidn2) public key set.
    private readonly IReadOnlyDictionary<string, byte[]>? _previousPublicKeys =
        LicensePublicKeyProvider.CurrentSnapshot();

    public LicenseBuildKeyFixture()
    {
        BuildKeyProvider.OverrideForTesting(LicenseTestSupport.TestBuildKey);
        LicensePublicKeyProvider.OverrideForTesting(LicenseTestSupport.DefaultTestKeySet());
    }

    public void Dispose()
    {
        BuildKeyProvider.OverrideForTesting(_previousKey);
        LicensePublicKeyProvider.OverrideForTesting(_previousPublicKeys);
    }
}

[CollectionDefinition("License", DisableParallelization = true)]
public sealed class LicenseCollection : Xunit.ICollectionFixture<LicenseBuildKeyFixture>
{
}
