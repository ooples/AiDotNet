using System.Security.Cryptography;
using System.Text;
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
        // The validate-license Edge Function expects:
        // { license_key: string, machine_id_hash: string, hostname?: string, os_description?: string }
        var licenseKey = "aidn.test12345678.abcdefghijklmnop";
        var machineIdHash = ComputeMachineIdHash();

        var requestBody = new
        {
            license_key = licenseKey,
            machine_id_hash = machineIdHash,
        };

        var json = JsonSerializer.Serialize(requestBody);
        var parsed = JsonDocument.Parse(json);

        Assert.True(parsed.RootElement.TryGetProperty("license_key", out var keyProp));
        Assert.Equal(licenseKey, keyProp.GetString());
        Assert.True(parsed.RootElement.TryGetProperty("machine_id_hash", out var hashProp));
        Assert.NotNull(hashProp.GetString());
        Assert.NotEmpty(hashProp.GetString()!);
    }

    [Fact]
    public void MachineIdHash_IsDeterministic()
    {
        var hash1 = ComputeMachineIdHash();
        var hash2 = ComputeMachineIdHash();
        Assert.Equal(hash1, hash2);
    }

    [Fact]
    public void MachineIdHash_IsSha256Format()
    {
        var hash = ComputeMachineIdHash();
        // SHA-256 hex string is 64 characters
        Assert.Equal(64, hash.Length);
        Assert.Matches("^[0-9a-f]{64}$", hash);
    }

    // ─── Response parsing tests ───

    [Fact]
    public void ParseValidResponse_ActiveLicense()
    {
        var responseJson = JsonSerializer.Serialize(new
        {
            valid = true,
            tier = "professional",
            message = "License is valid.",
        });

        var result = ParseValidateResponse(responseJson, 200);

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

        var result = ParseValidateResponse(responseJson, 200);

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

        var result = ParseValidateResponse(responseJson, 200);

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

        var result = ParseValidateResponse(responseJson, 400);

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

        var result = ParseValidateResponse(responseJson, 403);

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

        var result = ParseValidateResponse(responseJson, 403);

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

        var result = ParseValidateResponse(responseJson, 403);

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

        var result = ParseValidateResponse(responseJson, 403);

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

        var result = ParseValidateResponse(responseJson, 500);

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

        var result = ParseValidateResponse(responseJson, 400);

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

        var result = ParseValidateResponse(responseJson, 422);

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

        var result = ParseValidateResponse(responseJson, 502);

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

    [Fact]
    public void CommunityLicenseResponse_NewKey_HasCorrectShape()
    {
        var responseJson = JsonSerializer.Serialize(new
        {
            success = true,
            license_key = "aidn.abc123def456.ghijklmnopqrstuv",
            tier = "community",
            is_existing = false,
            message = "Community license created successfully!",
        });

        var doc = JsonDocument.Parse(responseJson);
        var root = doc.RootElement;

        Assert.True(root.GetProperty("success").GetBoolean());
        Assert.StartsWith("aidn.", root.GetProperty("license_key").GetString());
        Assert.Equal("community", root.GetProperty("tier").GetString());
        Assert.False(root.GetProperty("is_existing").GetBoolean());
    }

    [Fact]
    public void CommunityLicenseResponse_ExistingKey_HasCorrectShape()
    {
        var responseJson = JsonSerializer.Serialize(new
        {
            success = true,
            license_key = "aidn.existing1234.abcdefghijklmnop",
            tier = "community",
            is_existing = true,
            message = "You already have an active community license.",
        });

        var doc = JsonDocument.Parse(responseJson);
        var root = doc.RootElement;

        Assert.True(root.GetProperty("success").GetBoolean());
        Assert.True(root.GetProperty("is_existing").GetBoolean());
    }

    [Fact]
    public void CommunityLicenseResponse_Error_HasCorrectShape()
    {
        var responseJson = JsonSerializer.Serialize(new
        {
            success = false,
            error = "unauthorized",
            message = "Authentication required.",
        });

        var doc = JsonDocument.Parse(responseJson);
        var root = doc.RootElement;

        Assert.False(root.GetProperty("success").GetBoolean());
        Assert.Equal("unauthorized", root.GetProperty("error").GetString());
    }

    // ─── Stripe webhook contract tests ───

    [Fact]
    public void StripeWebhook_LicenseKeyFormat_IsValid()
    {
        // The webhook generates keys in format: aidn.{12chars}.{16chars}
        var keyPart1 = Guid.NewGuid().ToString("N")[..12];
        var keyPart2 = Guid.NewGuid().ToString("N")[..16];
        var key = $"aidn.{keyPart1}.{keyPart2}";

        Assert.StartsWith("aidn.", key);
        var parts = key.Split('.');
        Assert.Equal(3, parts.Length);
        Assert.Equal("aidn", parts[0]);
        Assert.Equal(12, parts[1].Length);
        Assert.Equal(16, parts[2].Length);
    }

    // ─── Helpers ───

    private static string ComputeMachineIdHash()
    {
        var machineFingerprint = Environment.MachineName + "|" + Environment.UserName;
        var saltedInput = "license-validation:" + machineFingerprint;
        using var sha256 = SHA256.Create();
        var bytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(saltedInput));
        return BitConverter.ToString(bytes).Replace("-", "").ToLowerInvariant();
    }

    /// <summary>
    /// Mimics LicenseValidator.ParseResponse logic to test the contract mapping.
    /// </summary>
    private static (LicenseKeyStatus Status, string? Tier) ParseValidateResponse(string responseJson, int httpStatusCode)
    {
        using var doc = JsonDocument.Parse(responseJson);
        var root = doc.RootElement;

        if (root.TryGetProperty("valid", out var validProp) && validProp.GetBoolean())
        {
            var tier = root.TryGetProperty("tier", out var tierProp) ? tierProp.GetString() : null;
            return (LicenseKeyStatus.Active, tier);
        }

        var error = root.TryGetProperty("error", out var errorProp) ? errorProp.GetString() : null;
        var status = error switch
        {
            "invalid_key" => LicenseKeyStatus.Invalid,
            "license_expired" => LicenseKeyStatus.Expired,
            "license_revoked" => LicenseKeyStatus.Revoked,
            "license_suspended" => LicenseKeyStatus.Revoked,
            "activation_limit" => LicenseKeyStatus.SeatLimitReached,
            "server_error" => LicenseKeyStatus.ValidationPending,
            "missing_fields" => LicenseKeyStatus.Invalid,
            "method_not_allowed" => LicenseKeyStatus.Invalid,
            _ => httpStatusCode >= 500 ? LicenseKeyStatus.ValidationPending : LicenseKeyStatus.Invalid
        };

        return (status, null);
    }
}
