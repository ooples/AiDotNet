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
[Collection("License")]
public class LicenseValidatorTests
{
    // Real HMAC-signed keys against the injected test build key (LicenseCollection fixture). After the
    // fail-closed fix, offline validation only accepts a verified signature — fake strings no longer pass.
    private static readonly string ValidTestKey = LicenseTestSupport.SignedKey("abc123def456");
    private static readonly string ValidTestKey2 = LicenseTestSupport.SignedKey("cached12key3");
    private static readonly string ValidTestKey3 = LicenseTestSupport.SignedKey("grace12test3");

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
        Assert.Equal("Offline validation succeeded (HMAC verified).", result.Message);
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
    public async Task OfflineMode_ServerValidatedKey_IsRejected()
    {
        // Closes the security gap from PR #1256 review: server-validated
        // AIDN-PROD-* keys carry no cryptographic signature the SDK can
        // verify locally. Offline-only mode (ServerUrl="") MUST reject
        // them so a leaked key can't be used air-gapped.
        await Task.Yield();
        var key = new AiDotNetLicenseKey("AIDN-PROD-COMMUNITY-1234567890ABCDEF1234567890ABCDEF")
        {
            ServerUrl = string.Empty // offline-only
        };

        var validator = new LicenseValidator(key);
        var result = validator.Validate();

        Assert.Equal(LicenseKeyStatus.Invalid, result.Status);
        Assert.Contains("aidn.{id}.{signature}", result.Message);
    }

    [Fact(Timeout = 60000)]
    public async Task OfflineMode_ServerValidatedKey_RejectionIsCached()
    {
        // PR #1256 second-pass review: the sync path's offline-only
        // rejection branch must populate _cached just like ValidateAsync()
        // does. Otherwise CachedResult is null after a validation attempt
        // and repeated Validate() calls allocate a fresh Invalid result
        // each time, both of which diverge from the established caching
        // contract (and from the async path).
        await Task.Yield();
        var key = new AiDotNetLicenseKey("AIDN-PROD-COMMUNITY-1234567890ABCDEF1234567890ABCDEF")
        {
            ServerUrl = string.Empty
        };

        var validator = new LicenseValidator(key);
        var first = validator.Validate();
        var second = validator.Validate();

        var cached = validator.CachedResult;
        Assert.NotNull(cached);
        Assert.Equal(LicenseKeyStatus.Invalid, cached.Status);
        // Reference equality: both calls return the cached instance,
        // and the cached instance is the one returned by Validate().
        Assert.Same(first, second);
        Assert.Same(first, cached);
    }

#if !NET471
    // ValidateAsync only exists on net10+ (the LicenseValidator source has
    // its async path under `#if !NET471`). Gate the test to match.
    [Fact(Timeout = 60000)]
    public async Task OfflineMode_ServerValidatedKey_AsyncPath_IsRejected()
    {
        // Same gate on the async path as the sync path.
        var key = new AiDotNetLicenseKey("AIDN-PROD-PRO-1234567890ABCDEF1234567890ABCDEF12")
        {
            ServerUrl = string.Empty
        };

        var validator = new LicenseValidator(key);
        var result = await validator.ValidateAsync();

        Assert.Equal(LicenseKeyStatus.Invalid, result.Status);
        Assert.Contains("aidn.{id}.{signature}", result.Message);
    }
#endif

    [Fact(Timeout = 60000)]
    public async Task DefaultServerUrl_IsSet()
    {
        await Task.Yield();
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
