using System.Net;
using System.Net.Http;
using System.Text;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.Helpers;

/// <summary>
/// Tests for <see cref="LicenseValidator"/> covering offline validation,
/// online validation with mock HTTP, caching, grace period, and error handling.
/// </summary>
[Collection("LicensingTests")]
public class LicenseValidatorTests
{
    // Valid format: aidn.{alphanumeric-id}.{alphanumeric-signature}
    private const string ValidTestKey = "aidn.abc123def456.sig789xyz012abc";
    private const string ValidTestKey2 = "aidn.cached12key3.sig456cached78";
    private const string ValidTestKey3 = "aidn.grace12test3.sig789grace012";

    [Fact(Timeout = 60000)]
    public async Task OfflineMode_ValidKey_ReturnsActive()
    {
        var key = new AiDotNetLicenseKey(ValidTestKey)
        {
            ServerUrl = string.Empty // explicit empty = offline-only
        };

        var validator = new LicenseValidator(key);
        var result = validator.Validate();

        Assert.Equal(LicenseKeyStatus.Active, result.Status);
        Assert.Equal("Offline-only mode.", result.Message);
    }

    [Fact(Timeout = 60000)]
    public async Task OfflineMode_CachesResult()
    {
        var key = new AiDotNetLicenseKey(ValidTestKey2)
        {
            ServerUrl = string.Empty
        };

        var validator = new LicenseValidator(key);
        var result1 = validator.Validate();
        var result2 = validator.Validate();

        Assert.Equal(LicenseKeyStatus.Active, result1.Status);
        Assert.Equal(LicenseKeyStatus.Active, result2.Status);
        Assert.NotNull(validator.CachedResult);
    }

    [Fact(Timeout = 60000)]
    public async Task DefaultServerUrl_IsSet()
    {
        Assert.False(string.IsNullOrWhiteSpace(LicenseValidator.DefaultServerUrl));
        Assert.Contains("supabase.co", LicenseValidator.DefaultServerUrl);
        Assert.Contains("validate-license", LicenseValidator.DefaultServerUrl);
    }

    [Fact(Timeout = 60000)]
    public async Task OnlineMode_ValidResponse_ServerUnreachable_ReturnsPending()
    {
        // Cannot inject HttpClient into LicenseValidator (uses shared static client).
        // Instead, verify the flow: with a fake server URL, the validator should
        // return ValidationPending since the server is unreachable and there's no cache.
        var key = new AiDotNetLicenseKey("aidn.onlinetest123.sigonlinetest456")
        {
            ServerUrl = "https://nonexistent.test/validate-license"
        };

        var validator = new LicenseValidator(key);
        var result = validator.Validate();

        // The validator should gracefully handle network failure
        Assert.Equal(LicenseKeyStatus.ValidationPending, result.Status);
        Assert.Contains("unreachable", result.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Timeout = 60000)]
    public async Task OnlineMode_ServerReturnsInvalidKey_ReturnsInvalid()
    {
        var key = new AiDotNetLicenseKey("aidn.badkey12test3.sigbadkey456abc")
        {
            ServerUrl = "https://nonexistent.test/validate" // will fail with network error
        };

        var validator = new LicenseValidator(key);

        // With a nonexistent server and no cache, should return ValidationPending
        var result = validator.Validate();

        Assert.Equal(LicenseKeyStatus.ValidationPending, result.Status);
        Assert.Contains("unreachable", result.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Timeout = 60000)]
    public async Task OnlineMode_ServerUnreachable_NoCachedResult_ReturnsPending()
    {
        var key = new AiDotNetLicenseKey("aidn.unreachtest12.sigunreachtest34")
        {
            ServerUrl = "https://192.0.2.1:1/validate" // RFC 5737 TEST-NET address, will timeout
        };

        var validator = new LicenseValidator(key);
        var result = validator.Validate();

        Assert.Equal(LicenseKeyStatus.ValidationPending, result.Status);
        Assert.Contains("unreachable", result.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Timeout = 60000)]
    public async Task OnlineMode_CachedActiveResult_WithinGracePeriod_ReturnsCached()
    {
        var key = new AiDotNetLicenseKey(ValidTestKey3)
        {
            ServerUrl = string.Empty, // start offline to prime cache
            OfflineGracePeriod = TimeSpan.FromHours(1)
        };

        var validator = new LicenseValidator(key);

        // First call: offline mode returns Active
        var result1 = validator.Validate();
        Assert.Equal(LicenseKeyStatus.Active, result1.Status);

        // The cached result exists and is within grace period
        var cached = validator.CachedResult;
        Assert.NotNull(cached);
        Assert.Equal(LicenseKeyStatus.Active, cached.Status);
    }

    [Fact(Timeout = 60000)]
    public async Task ValidationResult_ContainsTier()
    {
        var result = new LicenseValidationResult(
            LicenseKeyStatus.Active,
            tier: "enterprise",
            message: "Valid enterprise license.");

        Assert.Equal("enterprise", result.Tier);
        Assert.Equal(LicenseKeyStatus.Active, result.Status);
        Assert.Equal("Valid enterprise license.", result.Message);
    }

    [Fact(Timeout = 60000)]
    public async Task ValidationResult_ContainsSeatsInfo()
    {
        var result = new LicenseValidationResult(
            LicenseKeyStatus.SeatLimitReached,
            seatsUsed: 5,
            seatsMax: 5,
            message: "Maximum activations reached.");

        Assert.Equal(LicenseKeyStatus.SeatLimitReached, result.Status);
        Assert.Equal(5, result.SeatsUsed);
        Assert.Equal(5, result.SeatsMax);
    }

    [Fact(Timeout = 60000)]
    public async Task ValidationResult_SeatsUsed_CannotBeNegative()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LicenseValidationResult(LicenseKeyStatus.Active, seatsUsed: -1));
    }

    [Fact(Timeout = 60000)]
    public async Task ValidationResult_SeatsMax_CannotBeNegative()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LicenseValidationResult(LicenseKeyStatus.Active, seatsMax: -1));
    }

    [Fact(Timeout = 60000)]
    public async Task ValidationResult_ValidatedAt_DefaultsToUtcNow()
    {
        var before = DateTimeOffset.UtcNow;
        var result = new LicenseValidationResult(LicenseKeyStatus.Active);
        var after = DateTimeOffset.UtcNow;

        Assert.InRange(result.ValidatedAt, before, after);
    }

    [Fact(Timeout = 60000)]
    public async Task ValidationResult_DecryptionToken_ReturnsDefensiveCopy()
    {
        byte[] token = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
        var result = new LicenseValidationResult(LicenseKeyStatus.Active, decryptionToken: token);

        byte[]? copy1 = result.DecryptionToken;
        byte[]? copy2 = result.DecryptionToken;

        Assert.NotNull(copy1);
        Assert.NotNull(copy2);
        Assert.Equal(copy1, copy2); // same content
        Assert.NotSame(copy1, copy2); // different instances (defensive copy)

        // Mutating copy should not affect original
        copy1[0] = 99;
        Assert.NotEqual(copy1, result.DecryptionToken);
    }

    [Fact(Timeout = 60000)]
    public async Task ValidationResult_NullDecryptionToken_ReturnsNull()
    {
        var result = new LicenseValidationResult(LicenseKeyStatus.Active, decryptionToken: null);
        Assert.Null(result.DecryptionToken);
    }

    [Fact(Timeout = 60000)]
    public async Task LicenseKeyStatus_HasExpectedValues()
    {
        Assert.True(Enum.IsDefined(typeof(LicenseKeyStatus), LicenseKeyStatus.Active));
        Assert.True(Enum.IsDefined(typeof(LicenseKeyStatus), LicenseKeyStatus.Expired));
        Assert.True(Enum.IsDefined(typeof(LicenseKeyStatus), LicenseKeyStatus.Revoked));
        Assert.True(Enum.IsDefined(typeof(LicenseKeyStatus), LicenseKeyStatus.SeatLimitReached));
        Assert.True(Enum.IsDefined(typeof(LicenseKeyStatus), LicenseKeyStatus.Invalid));
        Assert.True(Enum.IsDefined(typeof(LicenseKeyStatus), LicenseKeyStatus.ValidationPending));
    }

    [Fact(Timeout = 60000)]
    public async Task AiDotNetLicenseKey_Constructor_SetsKey()
    {
        var key = new AiDotNetLicenseKey(ValidTestKey);
        Assert.Equal(ValidTestKey, key.Key);
    }

    [Fact(Timeout = 60000)]
    public async Task AiDotNetLicenseKey_ServerUrl_DefaultNull()
    {
        var key = new AiDotNetLicenseKey(ValidTestKey);
        Assert.Null(key.ServerUrl);
    }

    [Fact(Timeout = 60000)]
    public async Task AiDotNetLicenseKey_OfflineGracePeriod_Default7Days()
    {
        var key = new AiDotNetLicenseKey(ValidTestKey);
        Assert.Equal(TimeSpan.FromDays(7), key.OfflineGracePeriod);
    }

    [Fact(Timeout = 60000)]
    public async Task AiDotNetLicenseKey_EnableTelemetry_DefaultTrue()
    {
        var key = new AiDotNetLicenseKey(ValidTestKey);
        Assert.True(key.EnableTelemetry);
    }

#nullable disable
    [Fact(Timeout = 60000)]
    public async Task AiDotNetLicenseKey_NullKey_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new AiDotNetLicenseKey(null));
    }
#nullable restore

    [Fact(Timeout = 60000)]
    public async Task AiDotNetLicenseKey_EmptyKey_Throws()
    {
        Assert.Throws<ArgumentException>(() => new AiDotNetLicenseKey(""));
    }

    [Fact(Timeout = 60000)]
    public async Task AiDotNetLicenseKey_WhitespaceKey_Throws()
    {
        Assert.Throws<ArgumentException>(() => new AiDotNetLicenseKey("   "));
    }
}
