using System.Threading.Tasks;
namespace AiDotNet.Tests.UnitTests.Serialization;

using System;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models;
using Xunit;

public class LicenseKeyTests
{
    // ────────── AiDotNetLicenseKey construction ──────────

    [Fact(Timeout = 60000)]
    public async Task AiDotNetLicenseKey_ValidKey_SetsKeyProperty()
    {
        var license = new AiDotNetLicenseKey("aidn.abc123.secretXYZ");

        Assert.Equal("aidn.abc123.secretXYZ", license.Key);
    }

    [Fact(Timeout = 60000)]
    public async Task AiDotNetLicenseKey_NullKey_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new AiDotNetLicenseKey(null));
    }

    [Fact(Timeout = 60000)]
    public async Task AiDotNetLicenseKey_WhitespaceKey_Throws()
    {
        Assert.Throws<ArgumentException>(() => new AiDotNetLicenseKey("   "));
    }

    [Fact(Timeout = 60000)]
    public async Task AiDotNetLicenseKey_DefaultProperties_HaveCorrectValues()
    {
        var license = new AiDotNetLicenseKey("test-key");

        Assert.Null(license.ServerUrl);
        Assert.Null(license.Environment);
        Assert.Equal(TimeSpan.FromDays(7), license.OfflineGracePeriod);
        Assert.True(license.EnableTelemetry);
    }

    [Fact(Timeout = 60000)]
    public async Task AiDotNetLicenseKey_CustomProperties_AreSettable()
    {
        var license = new AiDotNetLicenseKey("test-key")
        {
            ServerUrl = "https://license.example.com",
            Environment = "staging",
            OfflineGracePeriod = TimeSpan.FromDays(14),
            EnableTelemetry = false
        };

        Assert.Equal("https://license.example.com", license.ServerUrl);
        Assert.Equal("staging", license.Environment);
        Assert.Equal(TimeSpan.FromDays(14), license.OfflineGracePeriod);
        Assert.False(license.EnableTelemetry);
    }

    // ────────── LicenseKeyResolver ──────────

    [Fact(Timeout = 60000)]
    public async Task LicenseKeyResolver_ExplicitKey_ReturnsThatKey()
    {
        var license = new AiDotNetLicenseKey("explicit-key-value");
        string? resolved = LicenseKeyResolver.Resolve(license);

        Assert.Equal("explicit-key-value", resolved);
    }

    [Fact(Timeout = 60000)]
    public async Task LicenseKeyResolver_NullLicense_FallsThrough()
    {
        // Save and clear env var to ensure clean state
        string? originalValue = System.Environment.GetEnvironmentVariable(LicenseKeyResolver.EnvVarName);
        System.Environment.SetEnvironmentVariable(LicenseKeyResolver.EnvVarName, null);
        try
        {
            string? resolved = LicenseKeyResolver.Resolve(null);

            // Without explicit key, env var, or license file, should return null
            Assert.Null(resolved);
        }
        finally
        {
            // Restore original value
            System.Environment.SetEnvironmentVariable(LicenseKeyResolver.EnvVarName, originalValue);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task LicenseKeyResolver_EnvVar_FallbackWhenNoExplicit()
    {
        string? originalValue = System.Environment.GetEnvironmentVariable(LicenseKeyResolver.EnvVarName);
        string testKey = "env-var-test-key-" + Guid.NewGuid().ToString("N");
        System.Environment.SetEnvironmentVariable(LicenseKeyResolver.EnvVarName, testKey);
        try
        {
            string? resolved = LicenseKeyResolver.Resolve(null);
            Assert.Equal(testKey, resolved);
        }
        finally
        {
            System.Environment.SetEnvironmentVariable(LicenseKeyResolver.EnvVarName, originalValue);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task LicenseKeyResolver_ExplicitKey_TakesPriorityOverEnvVar()
    {
        string? originalValue = System.Environment.GetEnvironmentVariable(LicenseKeyResolver.EnvVarName);
        System.Environment.SetEnvironmentVariable(LicenseKeyResolver.EnvVarName, "env-key");
        try
        {
            var license = new AiDotNetLicenseKey("explicit-key");
            string? resolved = LicenseKeyResolver.Resolve(license);
            Assert.Equal("explicit-key", resolved);
        }
        finally
        {
            System.Environment.SetEnvironmentVariable(LicenseKeyResolver.EnvVarName, originalValue);
        }
    }

    // ────────── MachineFingerprint ──────────

    [Fact(Timeout = 60000)]
    public async Task MachineFingerprint_GetMachineId_ReturnsNonEmptyString()
    {
        string id = MachineFingerprint.GetMachineId();

        Assert.False(string.IsNullOrWhiteSpace(id));
    }

    [Fact(Timeout = 60000)]
    public async Task MachineFingerprint_GetMachineId_IsConsistent()
    {
        string id1 = MachineFingerprint.GetMachineId();
        string id2 = MachineFingerprint.GetMachineId();

        Assert.Equal(id1, id2);
    }

    [Fact(Timeout = 60000)]
    public async Task MachineFingerprint_GetMachineId_IsHexEncoded()
    {
        string id = MachineFingerprint.GetMachineId();

        // SHA-256 hex = 64 chars
        Assert.Equal(64, id.Length);
        Assert.Matches("^[0-9a-f]{64}$", id);
    }

    // ────────── LicenseValidator (offline mode) ──────────

    [Fact(Timeout = 60000)]
    public async Task LicenseValidator_ExplicitOffline_ReturnsActive()
    {
        // Use empty ServerUrl for explicit offline-only mode.
        // Null ServerUrl now means "use DefaultServerUrl" (online validation).
        var license = new AiDotNetLicenseKey("aidn.test123.abc456")
        {
            ServerUrl = ""  // Explicit offline mode
        };
        var validator = new LicenseValidator(license);

        var result = validator.Validate();

        Assert.Equal(LicenseKeyStatus.Active, result.Status);
    }

    [Fact(Timeout = 60000)]
    public async Task LicenseValidator_ExplicitOffline_SetsOfflineMessage()
    {
        var license = new AiDotNetLicenseKey("aidn.test123.abc456")
        {
            ServerUrl = ""  // Explicit offline mode
        };
        var validator = new LicenseValidator(license);

        var result = validator.Validate();

        Assert.Contains("Offline", result.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Timeout = 60000)]
    public async Task LicenseValidator_UnreachableServer_ReturnsValidationPending()
    {
        var license = new AiDotNetLicenseKey("test-key")
        {
            ServerUrl = "http://127.0.0.1:1"  // Not a real server
        };
        var validator = new LicenseValidator(license);

        var result = validator.Validate();

        Assert.Equal(LicenseKeyStatus.ValidationPending, result.Status);
    }

    // ────────── LicenseValidationResult ──────────

    [Fact(Timeout = 60000)]
    public async Task LicenseValidationResult_DefaultsAreReasonable()
    {
        var result = new LicenseValidationResult(LicenseKeyStatus.Active);

        Assert.Equal(LicenseKeyStatus.Active, result.Status);
        Assert.Null(result.Tier);
        Assert.Null(result.ExpiresAt);
        Assert.Equal(0, result.SeatsUsed);
        Assert.Null(result.SeatsMax);
        Assert.True(result.ValidatedAt <= DateTimeOffset.UtcNow);
        Assert.Null(result.Message);
    }

    // ────────── AiModelBuilder integration ──────────

    [Fact(Timeout = 60000)]
    public async Task AiModelBuilder_DefaultConstructor_LicenseKeyIsNull()
    {
        var builder = new AiModelBuilder<double, double[], double>();

        Assert.Null(builder.ConfiguredLicenseKey);
    }

    [Fact(Timeout = 60000)]
    public async Task AiModelBuilder_ConstructorWithLicenseKey_StoresKey()
    {
        var license = new AiDotNetLicenseKey("test-key-123");
        var builder = new AiModelBuilder<double, double[], double>(license);

        Assert.NotNull(builder.ConfiguredLicenseKey);
        Assert.Equal("test-key-123", builder.ConfiguredLicenseKey.Key);
    }

    [Fact(Timeout = 60000)]
    public async Task AiModelBuilder_ConfigureLicenseKey_SetsKey()
    {
        var license = new AiDotNetLicenseKey("fluent-key");
        var builder = new AiModelBuilder<double, double[], double>();

        builder.ConfigureLicenseKey(license);

        Assert.NotNull(builder.ConfiguredLicenseKey);
        Assert.Equal("fluent-key", builder.ConfiguredLicenseKey.Key);
    }

    [Fact(Timeout = 60000)]
    public async Task AiModelBuilder_ConfigureLicenseKey_ReturnsSameBuilder()
    {
        var license = new AiDotNetLicenseKey("fluent-key");
        var builder = new AiModelBuilder<double, double[], double>();

        var returned = builder.ConfigureLicenseKey(license);

        Assert.Same(builder, returned);
    }

    [Fact(Timeout = 60000)]
    public async Task AiModelBuilder_ConfigureLicenseKey_NullThrows()
    {
        var builder = new AiModelBuilder<double, double[], double>();

        Assert.Throws<ArgumentNullException>(() => builder.ConfigureLicenseKey(null));
    }
}
