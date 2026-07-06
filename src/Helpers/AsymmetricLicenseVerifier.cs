using System.Security.Cryptography;
using AiDotNet.Enums;
using AiDotNet.Models;

namespace AiDotNet.Helpers;

/// <summary>
/// Verifies <c>aidn2.</c> asymmetric license tokens against the embedded PUBLIC signing key(s).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is the security core of the new licensing scheme. A token looks
/// like <c>aidn2.&lt;claims&gt;.&lt;signature&gt;</c>. We recompute nothing secret — we only check that
/// the signature was produced by the holder of the private key (which lives on the server and never
/// ships). Because verification uses a <b>public</b> key, an attacker who extracts it from the DLL
/// still cannot forge a valid token. This is strictly stronger than the old symmetric HMAC scheme,
/// where the same secret had to ship in every DLL and could be used to mint unlimited fake keys.</para>
///
/// <para><b>Primitive:</b> RSA-2048 (or larger) with RSASSA-PKCS1-v1.5 and SHA-256 — the exact
/// algorithm behind JWT <c>RS256</c>. Chosen because it is available in
/// <c>System.Security.Cryptography</c> on <b>every</b> target framework (net471/net8/net10) with no
/// added dependency, and importing a public key from modulus+exponent works uniformly across all of
/// them (unlike <c>ImportSubjectPublicKeyInfo</c>, which does not exist on net471, or Ed25519/PSS,
/// which are not available on net471's default RSA implementation).</para>
///
/// <para><b>Token grammar:</b> <c>aidn2.&lt;base64url(claims_json)&gt;.&lt;base64url(signature)&gt;</c>
/// where the signature is computed over the raw UTF-8 bytes of <c>claims_json</c> (i.e. the exact
/// bytes obtained by base64url-decoding the middle segment).</para>
/// </remarks>
internal static class AsymmetricLicenseVerifier
{
    /// <summary>The token version prefix for asymmetric (public-key-signed) licenses.</summary>
    internal const string Prefix = "aidn2";

    /// <summary>
    /// Allowed clock skew when checking expiry, so a token that is a few seconds past <c>exp</c>
    /// due to clock drift between the signing server and the client is not spuriously rejected.
    /// Kept small — expiry is the primary bound on offline use.
    /// </summary>
    private static readonly TimeSpan ClockSkew = TimeSpan.FromMinutes(5);

    /// <summary>
    /// Returns true if <paramref name="key"/> is in the asymmetric token shape:
    /// <c>aidn2.&lt;base64url&gt;.&lt;base64url&gt;</c>. Does NOT verify the signature — this is a
    /// cheap structural check used for routing (see <see cref="LicenseValidator"/>).
    /// </summary>
    internal static bool IsAsymmetricKeyFormat(string? key)
    {
        if (key is null) return false;
        var parts = key.Split('.');
        if (parts.Length != 3 || parts[0] != Prefix || parts[1].Length == 0 || parts[2].Length == 0)
        {
            return false;
        }

        for (int i = 0; i < parts[1].Length; i++)
        {
            if (!IsBase64UrlChar(parts[1][i])) return false;
        }

        for (int i = 0; i < parts[2].Length; i++)
        {
            if (!IsBase64UrlChar(parts[2][i])) return false;
        }

        return true;
    }

    private static bool IsBase64UrlChar(char c)
    {
        return (c >= 'A' && c <= 'Z')
            || (c >= 'a' && c <= 'z')
            || (c >= '0' && c <= '9')
            || c == '-'
            || c == '_';
    }

    /// <summary>
    /// Verifies an <c>aidn2.</c> token fully offline: checks the signature against the embedded public
    /// key selected by the token's <c>kid</c>, then enforces expiry. Fails closed on any problem.
    /// </summary>
    /// <param name="key">The full <c>aidn2.&lt;claims&gt;.&lt;sig&gt;</c> token.</param>
    /// <param name="nowUtc">The current UTC time (injectable for tests).</param>
    /// <returns>
    /// An <see cref="LicenseValidationResult"/>: <see cref="LicenseKeyStatus.Active"/> when the
    /// signature verifies and the token is unexpired; <see cref="LicenseKeyStatus.Expired"/> when the
    /// signature verifies but <c>exp</c> has passed; <see cref="LicenseKeyStatus.Invalid"/> for a bad
    /// signature, unknown <c>kid</c>, malformed token, or a build that embeds no public key.
    /// </returns>
    internal static LicenseValidationResult Verify(string key, DateTimeOffset nowUtc)
    {
        if (string.IsNullOrWhiteSpace(key))
        {
            return Invalid("License key is empty or missing.");
        }

        var parts = key.Split('.');
        if (parts.Length != 3 || parts[0] != Prefix)
        {
            return Invalid("License token is not a valid aidn2 token.");
        }

        byte[] claimsBytes;
        byte[] signature;
        try
        {
            claimsBytes = Base64UrlHelper.Decode(parts[1]);
            signature = Base64UrlHelper.Decode(parts[2]);
        }
        catch (FormatException)
        {
            return Invalid("License token encoding is malformed.");
        }

        if (claimsBytes.Length == 0 || signature.Length == 0)
        {
            return Invalid("License token is missing claims or signature.");
        }

        string claimsJson;
        try
        {
            claimsJson = System.Text.Encoding.UTF8.GetString(claimsBytes);
        }
        catch (ArgumentException)
        {
            return Invalid("License token claims are not valid UTF-8.");
        }

        LicenseClaims? claims = LicenseClaims.TryParse(claimsJson);
        if (claims is null)
        {
            return Invalid("License token claims are malformed.");
        }

        string? kid = claims.Kid;
        if (kid is null || kid.Length == 0)
        {
            return Invalid("License token does not name a signing key (kid).");
        }

        // A build that embeds no public key CANNOT verify a signature. It must NOT trust the token —
        // fail closed exactly like the HMAC path does when the build key is absent.
        if (!LicensePublicKeyProvider.TryGetPublicKey(kid, out RSAParameters publicKey))
        {
            return Invalid(
                "This build cannot verify the license signature: no embedded public key matches the " +
                "token's key id (kid='" + kid + "'). Use an official build, or online validation.");
        }

        bool signatureValid;
        try
        {
            using var rsa = RSA.Create();
            rsa.ImportParameters(publicKey);
            // RSASSA-PKCS1-v1.5 + SHA-256 == JWT RS256. Signature is over the raw claims bytes.
            signatureValid = rsa.VerifyData(
                claimsBytes, signature, HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1);
        }
        catch (CryptographicException ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "AsymmetricLicenseVerifier: verification error: " + ex.Message);
            return Invalid("License signature verification failed.");
        }

        if (!signatureValid)
        {
            return Invalid("License signature verification failed.");
        }

        // Signature is authentic. Now enforce expiry (bounds offline use of a leaked token).
        DateTimeOffset expiresAt = DateTimeOffset.FromUnixTimeSeconds(claims.Exp);
        if (claims.Exp > 0 && expiresAt + ClockSkew < nowUtc)
        {
            return new LicenseValidationResult(
                LicenseKeyStatus.Expired,
                tier: claims.Tier,
                expiresAt: expiresAt,
                seatsMax: claims.Seats > 0 ? claims.Seats : (int?)null,
                message: "License token expired on " + expiresAt.UtcDateTime.ToString("u") + ".");
        }

        return new LicenseValidationResult(
            LicenseKeyStatus.Active,
            tier: claims.Tier,
            expiresAt: claims.Exp > 0 ? expiresAt : (DateTimeOffset?)null,
            seatsMax: claims.Seats > 0 ? claims.Seats : (int?)null,
            message: "Offline validation succeeded (signature verified).");
    }

    private static LicenseValidationResult Invalid(string message) =>
        new(LicenseKeyStatus.Invalid, message: message);
}
