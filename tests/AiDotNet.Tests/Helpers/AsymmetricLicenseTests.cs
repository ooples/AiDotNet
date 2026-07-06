using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.Helpers;

/// <summary>
/// Regression tests for the asymmetric (<c>aidn2.</c>) license scheme: a valid signed token is Active
/// offline, a forged / tampered signature is rejected, an expired token is rejected, an unknown key-id is
/// rejected, and a build without the embedded public key fails closed. These prove the scheme is strictly
/// stronger than the retired symmetric HMAC path — a token cannot be minted without the private key, which
/// never ships. Legacy <c>aidn.</c> keys must keep validating (back-compat) — see LicenseValidatorTests.
/// </summary>
[Collection("License")]
public class AsymmetricLicenseTests
{
    // ───────────── structural / routing ─────────────

    [Fact(Timeout = 60000)]
    public async Task IsAsymmetricKeyFormat_RecognisesAidn2()
    {
        await Task.Yield();
        Assert.True(AsymmetricLicenseVerifier.IsAsymmetricKeyFormat(LicenseTestSupport.SignedKeyV2()));
        Assert.False(AsymmetricLicenseVerifier.IsAsymmetricKeyFormat(LicenseTestSupport.SignedKey("abc")));
        Assert.False(AsymmetricLicenseVerifier.IsAsymmetricKeyFormat("AIDN-PROD-PRO-1234567890ABCDEF"));
        Assert.False(AsymmetricLicenseVerifier.IsAsymmetricKeyFormat("aidn2.only-two-parts"));
        Assert.False(AsymmetricLicenseVerifier.IsAsymmetricKeyFormat(null));
    }

    [Fact(Timeout = 60000)]
    public async Task ValidateKeyFormat_AcceptsAidn2()
    {
        await Task.Yield();
        Assert.True(LicenseValidator.ValidateKeyFormat(LicenseTestSupport.SignedKeyV2()));
        Assert.True(LicenseValidator.IsOfflineVerifiableKeyFormat(LicenseTestSupport.SignedKeyV2()));
        // Server-validated keys are NOT offline-verifiable.
        Assert.False(LicenseValidator.IsOfflineVerifiableKeyFormat(
            "AIDN-PROD-COMMUNITY-1234567890ABCDEF1234567890ABCDEF"));
    }

    // ───────────── valid token ─────────────

    [Fact(Timeout = 60000)]
    public async Task Aidn2_ValidToken_OfflineMode_ReturnsActive()
    {
        await Task.Yield();
        var key = new AiDotNetLicenseKey(LicenseTestSupport.SignedKeyV2(tier: "enterprise", seats: 10))
        {
            ServerUrl = string.Empty // explicit offline-only
        };

        var result = new LicenseValidator(key).Validate();

        Assert.Equal(LicenseKeyStatus.Active, result.Status);
        Assert.Equal("enterprise", result.Tier);
        Assert.Equal(10, result.SeatsMax);
        Assert.Contains("signature verified", result.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Timeout = 60000)]
    public async Task Aidn2_ValidToken_DefaultServer_VerifiesOfflineWithoutNetwork()
    {
        // ServerUrl == null (default). Because a public key is embedded (fixture) and the token is aidn2,
        // the validator verifies offline and never touches the network — same opportunistic path as v1.
        await Task.Yield();
        var key = new AiDotNetLicenseKey(LicenseTestSupport.SignedKeyV2());

        var result = new LicenseValidator(key).Validate();

        Assert.Equal(LicenseKeyStatus.Active, result.Status);
    }

    // ───────────── forged / tampered ─────────────

    [Fact(Timeout = 60000)]
    public async Task Aidn2_ForgedSignature_ForeignKey_IsRejected()
    {
        // The core security property: a token signed by a keypair whose public half is NOT embedded
        // cannot be forged into validity. This is exactly what an attacker without the private key faces.
        await Task.Yield();
        using RSA foreign = LicenseTestSupport.CreateForeignSigningKey();
        string forged = LicenseTestSupport.SignedKeyV2(signingKey: foreign); // kid still = TestKid

        var key = new AiDotNetLicenseKey(forged) { ServerUrl = string.Empty };
        var result = new LicenseValidator(key).Validate();

        Assert.Equal(LicenseKeyStatus.Invalid, result.Status);
        Assert.Contains("signature", result.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Timeout = 60000)]
    public async Task Aidn2_TamperedClaims_IsRejected()
    {
        // Take a valid token and flip a byte in the claims segment: the signature no longer matches.
        await Task.Yield();
        string valid = LicenseTestSupport.SignedKeyV2(tier: "community");
        string[] parts = valid.Split('.');
        byte[] claims = Base64UrlHelper.Decode(parts[1]);
        // Upgrade "community" → "enterprise" would require re-signing; here just corrupt a byte.
        claims[claims.Length / 2] ^= 0xFF;
        string tampered = parts[0] + "." + Base64UrlHelper.Encode(claims) + "." + parts[2];

        var key = new AiDotNetLicenseKey(tampered) { ServerUrl = string.Empty };
        var result = new LicenseValidator(key).Validate();

        Assert.Equal(LicenseKeyStatus.Invalid, result.Status);
    }

    // ───────────── expired ─────────────

    [Fact(Timeout = 60000)]
    public async Task Aidn2_ExpiredToken_IsRejected()
    {
        await Task.Yield();
        string expired = LicenseTestSupport.SignedKeyV2(
            iat: DateTimeOffset.UtcNow.AddDays(-60),
            exp: DateTimeOffset.UtcNow.AddDays(-30)); // well past the clock-skew allowance

        var key = new AiDotNetLicenseKey(expired) { ServerUrl = string.Empty };
        var result = new LicenseValidator(key).Validate();

        Assert.Equal(LicenseKeyStatus.Expired, result.Status);
        Assert.Contains("expired", result.Message, StringComparison.OrdinalIgnoreCase);
    }

    // ───────────── unknown kid / no embedded key (fail closed) ─────────────

    [Fact(Timeout = 60000)]
    public async Task Aidn2_UnknownKid_IsRejected()
    {
        await Task.Yield();
        string token = LicenseTestSupport.SignedKeyV2(kid: "no-such-kid");

        var key = new AiDotNetLicenseKey(token) { ServerUrl = string.Empty };
        var result = new LicenseValidator(key).Validate();

        Assert.Equal(LicenseKeyStatus.Invalid, result.Status);
        Assert.Contains("kid", result.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Timeout = 60000)]
    public async Task Aidn2_NoEmbeddedPublicKey_FailsClosed()
    {
        // Simulate a dev/fork build with no embedded public key: even a genuinely-signed token cannot be
        // verified locally, so offline validation must reject it rather than fail open.
        await Task.Yield();
        string token = LicenseTestSupport.SignedKeyV2();

        using (LicenseTestSupport.WithLicensePublicKeys(null))
        {
            var key = new AiDotNetLicenseKey(token) { ServerUrl = string.Empty };
            var result = new LicenseValidator(key).Validate();

            Assert.Equal(LicenseKeyStatus.Invalid, result.Status);
        }
    }

    // ───────────── revocation is server-driven (online) ─────────────

    [Fact(Timeout = 60000)]
    public async Task Aidn2_CustomServer_RoutesOnline_NotOfflineActive()
    {
        // With an explicit custom ServerUrl the validator must go ONLINE (where the server can deny a
        // revoked key) rather than short-circuiting to offline Active. An unreachable server yields
        // ValidationPending — proving the token was NOT accepted purely offline.
        await Task.Yield();
        var key = new AiDotNetLicenseKey(LicenseTestSupport.SignedKeyV2())
        {
            ServerUrl = "https://192.0.2.1:1/validate" // RFC 5737 TEST-NET, unreachable
        };

        var result = new LicenseValidator(key).Validate();

        Assert.Equal(LicenseKeyStatus.ValidationPending, result.Status);
    }

    [Fact(Timeout = 60000)]
    public async Task Aidn2_RevokedByServer_ParseResponse_ReturnsRevoked()
    {
        // The server remains the source of truth for revocation: a revoked key returns license_revoked,
        // which the client maps to Revoked (unchanged by the asymmetric offline path).
        await Task.Yield();
        string json = Newtonsoft.Json.JsonConvert.SerializeObject(new { valid = false, error = "license_revoked" });

        var result = LicenseValidator.ParseResponse(json, 403);

        Assert.Equal(LicenseKeyStatus.Revoked, result.Status);
    }

    // ───────────── back-compat: v1 and v2 coexist ─────────────

    [Fact(Timeout = 60000)]
    public async Task Aidn1_LegacyHmacKey_StillValidatesOffline()
    {
        // Deprecation window: legacy aidn. HMAC keys keep working against the embedded build key.
        await Task.Yield();
        var v1 = new AiDotNetLicenseKey(LicenseTestSupport.SignedKey("legacy-customer-1"))
        {
            ServerUrl = string.Empty
        };

        var result = new LicenseValidator(v1).Validate();

        Assert.Equal(LicenseKeyStatus.Active, result.Status);
        Assert.Contains("HMAC verified", result.Message, StringComparison.Ordinal);
    }

    // ───────────── provider round-trip ─────────────

    [Fact(Timeout = 60000)]
    public async Task LicensePublicKeyProvider_ParsesJwkManifest()
    {
        await Task.Yield();
        RSAParameters pub = LicenseTestSupport.TestPublicKeyParameters;
        string json = Newtonsoft.Json.JsonConvert.SerializeObject(new
        {
            keys = new[]
            {
                new
                {
                    kid = "roundtrip",
                    n = Base64UrlHelper.Encode(pub.Modulus),
                    e = Base64UrlHelper.Encode(pub.Exponent)
                }
            }
        });

        var parsed = LicensePublicKeyProvider.Parse(json);

        Assert.NotNull(parsed);
        Assert.True(parsed!.ContainsKey("roundtrip"));
        Assert.Equal(pub.Modulus, parsed["roundtrip"].Modulus);
        Assert.Equal(pub.Exponent, parsed["roundtrip"].Exponent);
    }
}
