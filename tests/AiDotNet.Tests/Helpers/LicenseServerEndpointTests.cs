using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.Helpers;

/// <summary>
/// Integration-style tests for the license server endpoint contract.
/// Tests validate the LicenseValidator against the expected Edge Function
/// request/response format without requiring a live server.
/// </summary>
[Collection("LicensingTests")]
public class LicenseServerEndpointTests
{
    // ─── Request format tests ───

    [Fact]
    public void ValidateRequest_ContainsRequiredFields()
    {
        // Exercise the ACTUAL production BuildRequestBody method
        var licenseKey = "aidn.test12345678.abcdefghijklmnop";
        var validator = new LicenseValidator(new AiDotNetLicenseKey(licenseKey)
        {
            ServerUrl = string.Empty,
            EnableTelemetry = false
        });

        var requestBody = validator.BuildRequestBody();

        var json = JsonSerializer.Serialize(requestBody);
        var parsed = JsonDocument.Parse(json);

        Assert.True(parsed.RootElement.TryGetProperty("license_key", out var keyProp));
        Assert.Equal(licenseKey, keyProp.GetString());
        Assert.True(parsed.RootElement.TryGetProperty("machine_id_hash", out var hashProp));
        // Verify the hash matches the production GetMachineIdHash output
        Assert.Equal(LicenseValidator.GetMachineIdHash(), hashProp.GetString());
        Assert.Matches("^[0-9a-f]{64}$", hashProp.GetString());
    }

    [Fact]
    public void MachineIdHash_IsDeterministic()
    {
        var hash1 = LicenseValidator.GetMachineIdHash();
        var hash2 = LicenseValidator.GetMachineIdHash();
        Assert.Equal(hash1, hash2);
    }

    [Fact]
    public void MachineIdHash_IsSha256Format()
    {
        var hash = LicenseValidator.GetMachineIdHash();
        // SHA-256 hex string is 64 characters
        Assert.Equal(64, hash.Length);
        Assert.Matches("^[0-9a-f]{64}$", hash);
    }

    // ─── Response parsing tests (using production ParseResponse) ───

    [Fact]
    public void ParseValidResponse_ActiveLicense()
    {
        var responseJson = JsonSerializer.Serialize(new
        {
            valid = true,
            tier = "professional",
            message = "License is valid.",
        });

        var result = LicenseValidator.ParseResponse(responseJson, 200);

        Assert.Equal(LicenseKeyStatus.Active, result.Status);
        Assert.Equal("professional", result.Tier);
    }

    [Fact]
    public void ParseValidResponse_CommunityLicense()
    {
        var responseJson = JsonSerializer.Serialize(new
        {
            valid = true,
            tier = "community",
            message = "License is valid.",
        });

        var result = LicenseValidator.ParseResponse(responseJson, 200);

        Assert.Equal(LicenseKeyStatus.Active, result.Status);
        Assert.Equal("community", result.Tier);
    }

    [Fact]
    public void ParseValidResponse_EnterpriseLicense()
    {
        var responseJson = JsonSerializer.Serialize(new
        {
            valid = true,
            tier = "enterprise",
            message = "License is valid.",
        });

        var result = LicenseValidator.ParseResponse(responseJson, 200);

        Assert.Equal(LicenseKeyStatus.Active, result.Status);
        Assert.Equal("enterprise", result.Tier);
    }

    [Fact]
    public void ParseErrorResponse_InvalidKey()
    {
        var responseJson = JsonSerializer.Serialize(new
        {
            valid = false,
            error = "invalid_key",
            message = "License key not found.",
        });

        var result = LicenseValidator.ParseResponse(responseJson, 400);

        Assert.Equal(LicenseKeyStatus.Invalid, result.Status);
    }

    [Fact]
    public void ParseErrorResponse_ExpiredLicense()
    {
        var responseJson = JsonSerializer.Serialize(new
        {
            valid = false,
            error = "license_expired",
            message = "License has expired.",
        });

        var result = LicenseValidator.ParseResponse(responseJson, 403);

        Assert.Equal(LicenseKeyStatus.Expired, result.Status);
    }

    [Fact]
    public void ParseErrorResponse_RevokedLicense()
    {
        var responseJson = JsonSerializer.Serialize(new
        {
            valid = false,
            error = "license_revoked",
            message = "License has been revoked.",
        });

        var result = LicenseValidator.ParseResponse(responseJson, 403);

        Assert.Equal(LicenseKeyStatus.Revoked, result.Status);
    }

    [Fact]
    public void ParseErrorResponse_SuspendedLicense()
    {
        var responseJson = JsonSerializer.Serialize(new
        {
            valid = false,
            error = "license_suspended",
            message = "License is suspended.",
        });

        var result = LicenseValidator.ParseResponse(responseJson, 403);

        Assert.Equal(LicenseKeyStatus.Revoked, result.Status); // Suspended maps to Revoked
    }

    [Fact]
    public void ParseErrorResponse_ActivationLimit()
    {
        var responseJson = JsonSerializer.Serialize(new
        {
            valid = false,
            error = "activation_limit",
            message = "Maximum machine activations reached.",
        });

        var result = LicenseValidator.ParseResponse(responseJson, 403);

        Assert.Equal(LicenseKeyStatus.SeatLimitReached, result.Status);
    }

    [Fact]
    public void ParseErrorResponse_ServerError_MapsToValidationPending()
    {
        var responseJson = JsonSerializer.Serialize(new
        {
            valid = false,
            error = "server_error",
            message = "Internal server error.",
        });

        var result = LicenseValidator.ParseResponse(responseJson, 500);

        Assert.Equal(LicenseKeyStatus.ValidationPending, result.Status);
    }

    [Fact]
    public void ParseErrorResponse_MissingFields()
    {
        var responseJson = JsonSerializer.Serialize(new
        {
            valid = false,
            error = "missing_fields",
            message = "Required fields are missing.",
        });

        var result = LicenseValidator.ParseResponse(responseJson, 400);

        Assert.Equal(LicenseKeyStatus.Invalid, result.Status);
    }

    [Fact]
    public void ParseErrorResponse_UnknownError_With4xx()
    {
        var responseJson = JsonSerializer.Serialize(new
        {
            valid = false,
            error = "some_new_error_type",
            message = "Something unexpected.",
        });

        var result = LicenseValidator.ParseResponse(responseJson, 422);

        Assert.Equal(LicenseKeyStatus.Invalid, result.Status);
    }

    [Fact]
    public void ParseErrorResponse_UnknownError_With5xx()
    {
        var responseJson = JsonSerializer.Serialize(new
        {
            valid = false,
            error = "some_unknown_error",
            message = "Server trouble.",
        });

        var result = LicenseValidator.ParseResponse(responseJson, 502);

        Assert.Equal(LicenseKeyStatus.ValidationPending, result.Status);
    }

    // ─── LicenseValidator integration with fake server ───

    [Fact]
    public void LicenseValidator_DefaultServerUrl_IsConfigured()
    {
        // Verify the default server URL points to the Supabase Edge Function
        Assert.Contains("supabase.co/functions/v1/validate-license", LicenseValidator.DefaultServerUrl);
    }

    [Fact]
    public void LicenseValidator_OfflineMode_ValidatesFormat()
    {
        // Offline mode (ServerUrl = "") should validate key format only
        var key = new AiDotNetLicenseKey("aidn.test12345678.abcdefghijklmnop")
        {
            ServerUrl = string.Empty,
        };

        var validator = new LicenseValidator(key);
        var result = validator.Validate();

        if (BuildKeyProvider.IsOfficialBuild)
        {
            // With a build key present, unsigned test keys fail HMAC verification
            Assert.Equal(LicenseKeyStatus.Invalid, result.Status);
        }
        else
        {
            // Without a build key, format-valid keys pass offline validation
            Assert.Equal(LicenseKeyStatus.Active, result.Status);
        }
    }

    [Fact]
    public void LicenseValidator_OfflineMode_InvalidFormat_RejectsKey()
    {
        // A key with invalid format should fail offline validation
        var key = new AiDotNetLicenseKey("invalid-key-format-no-aidn-prefix")
        {
            ServerUrl = string.Empty,
        };

        var validator = new LicenseValidator(key);
        var result = validator.Validate();

        Assert.Equal(LicenseKeyStatus.Invalid, result.Status);
    }

    [Fact]
    public void LicenseValidator_UnreachableServer_FallsBackGracefully()
    {
        // With a non-existent server, validation should fall back to offline mode
        var key = new AiDotNetLicenseKey("aidn.test12345678.abcdefghijklmnop")
        {
            ServerUrl = "https://localhost:1/nonexistent",
        };

        var validator = new LicenseValidator(key);
        var result = validator.Validate();

        // Should not crash — returns ValidationPending or falls back to offline
        Assert.NotNull(result);
        Assert.True(
            result.Status == LicenseKeyStatus.ValidationPending ||
            result.Status == LicenseKeyStatus.Active,
            $"Expected ValidationPending or Active (offline fallback), got {result.Status}");
    }

    [Fact]
    public void LicenseValidator_CachesResults()
    {
        var key = new AiDotNetLicenseKey("aidn.test12345678.abcdefghijklmnop")
        {
            ServerUrl = string.Empty,
        };

        var validator = new LicenseValidator(key);
        var result1 = validator.Validate();
        var result2 = validator.Validate();

        // Both calls should return the same result (cached)
        Assert.Equal(result1.Status, result2.Status);
    }

    // ─── Register community license Edge Function contract tests ───
    // NOTE: These tests validate the expected JSON contract from the Edge Function
    // (written in TypeScript). Since there is no C# production code producing these
    // responses, they test the contract schema that the C# client expects to consume.

    [Fact]
    public void CommunityLicenseResponse_NewKey_ParsedCorrectly()
    {
        // Simulate the JSON response the Edge Function returns and parse it
        // through the production ParseResponse to verify the mapping is correct
        var responseJson = JsonSerializer.Serialize(new
        {
            valid = true,
            tier = "community",
            message = "Community license created successfully!",
        });

        // Verify production code can parse this response correctly
        var result = LicenseValidator.ParseResponse(responseJson, 200);
        Assert.Equal(LicenseKeyStatus.Active, result.Status);
        Assert.Equal("community", result.Tier);
    }

    [Fact]
    public void CommunityLicenseResponse_ExistingKey_ParsedCorrectly()
    {
        var responseJson = JsonSerializer.Serialize(new
        {
            valid = true,
            tier = "community",
            message = "You already have an active community license.",
        });

        var result = LicenseValidator.ParseResponse(responseJson, 200);
        Assert.Equal(LicenseKeyStatus.Active, result.Status);
        Assert.Equal("community", result.Tier);
    }

    [Fact]
    public void CommunityLicenseResponse_Error_ParsedCorrectly()
    {
        var responseJson = JsonSerializer.Serialize(new
        {
            valid = false,
            error = "unauthorized",
            message = "Authentication required.",
        });

        // "unauthorized" is an unknown error type with 401 (4xx) → should map to Invalid
        var result = LicenseValidator.ParseResponse(responseJson, 401);
        Assert.Equal(LicenseKeyStatus.Invalid, result.Status);
    }

    // ─── Stripe webhook contract tests ───

    [Fact]
    public void StripeWebhook_LicenseKeyFormat_IsValid()
    {
        // The webhook generates keys in format: aidn.{12chars}.{16chars}
        // Validate that the generated format passes production key format validation
        var keyPart1 = Guid.NewGuid().ToString("N")[..12];
        var keyPart2 = Guid.NewGuid().ToString("N")[..16];
        var key = $"aidn.{keyPart1}.{keyPart2}";

        // Validate through production format checker, not hand-rolled checks
        Assert.True(LicenseValidator.ValidateKeyFormat(key),
            $"Generated key '{key}' should pass production format validation");

        // Also verify the parts structure
        var parts = key.Split('.');
        Assert.Equal(3, parts.Length);
        Assert.Equal("aidn", parts[0]);
    }

    // ─── BuildRequestBody tests ───

    [Fact]
    public void BuildRequestBody_WithTelemetryEnabled_IncludesHostname()
    {
        var validator = new LicenseValidator(new AiDotNetLicenseKey("aidn.test12345678.abcdefghijklmnop")
        {
            ServerUrl = string.Empty,
            EnableTelemetry = true
        });

        var body = validator.BuildRequestBody();

        Assert.True(body.ContainsKey("license_key"));
        Assert.True(body.ContainsKey("machine_id_hash"));
        // With telemetry enabled, hostname should be present
        Assert.True(body.ContainsKey("hostname"));
    }

    [Fact]
    public void BuildRequestBody_WithTelemetryDisabled_ExcludesHostname()
    {
        var validator = new LicenseValidator(new AiDotNetLicenseKey("aidn.test12345678.abcdefghijklmnop")
        {
            ServerUrl = string.Empty,
            EnableTelemetry = false
        });

        var body = validator.BuildRequestBody();

        Assert.True(body.ContainsKey("license_key"));
        Assert.True(body.ContainsKey("machine_id_hash"));
        // With telemetry disabled, hostname should NOT be present
        Assert.False(body.ContainsKey("hostname"));
    }
}
