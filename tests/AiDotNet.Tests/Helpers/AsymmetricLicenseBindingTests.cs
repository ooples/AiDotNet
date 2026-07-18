using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.Helpers;

/// <summary>
/// Tests the v2 aidn2 lockdown additions: scope/audience binding, machine (node-lock) binding, signed-CRL
/// revocation, and that the token's signed capabilities flow into the validation result. All run against
/// the real <see cref="AsymmetricLicenseVerifier"/> with the fixture-injected test public key, offline-only.
/// </summary>
[Collection("License")]
public class AsymmetricLicenseBindingTests
{
    private static LicenseValidationResult Validate(string key) =>
        new LicenseValidator(new AiDotNetLicenseKey(key) { ServerUrl = string.Empty }).Validate();

    // ───────────── caps flow-through ─────────────

    [Fact(Timeout = 60000)]
    public async Task Aidn2_Caps_AreCarriedIntoResult()
    {
        await Task.Yield();
        var result = Validate(LicenseTestSupport.SignedKeyV2(caps: new[] { "model:save", "tensors:save" }));

        Assert.Equal(LicenseKeyStatus.Active, result.Status);
        Assert.True(result.HasCapability("model:save"));
        Assert.True(result.HasCapability("tensors:save"));
        Assert.False(result.HasCapability("model:encrypt"));
    }

    // ───────────── scope binding ─────────────

    [Fact(Timeout = 60000)]
    public async Task Aidn2_Scope_RejectedWhenHostScopeUnset()
    {
        await Task.Yield();
        using var _ = new ScopeEnv(null);
        var result = Validate(LicenseTestSupport.SignedKeyV2(scope: "ci"));
        Assert.Equal(LicenseKeyStatus.Invalid, result.Status);
        Assert.Contains("scope", result.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Timeout = 60000)]
    public async Task Aidn2_Scope_RejectedWhenHostScopeDiffers()
    {
        await Task.Yield();
        using var _ = new ScopeEnv("prod");
        var result = Validate(LicenseTestSupport.SignedKeyV2(scope: "ci"));
        Assert.Equal(LicenseKeyStatus.Invalid, result.Status);
    }

    [Fact(Timeout = 60000)]
    public async Task Aidn2_Scope_AcceptedWhenHostScopeMatches()
    {
        await Task.Yield();
        using var _ = new ScopeEnv("ci");
        var result = Validate(LicenseTestSupport.SignedKeyV2(scope: "ci"));
        Assert.Equal(LicenseKeyStatus.Active, result.Status);
    }

    [Fact(Timeout = 60000)]
    public async Task Aidn2_NoScopeClaim_UnaffectedByHostScope()
    {
        await Task.Yield();
        using var _ = new ScopeEnv("something");
        var result = Validate(LicenseTestSupport.SignedKeyV2()); // no scope claim
        Assert.Equal(LicenseKeyStatus.Active, result.Status);
    }

    // ───────────── machine binding ─────────────

    [Fact(Timeout = 60000)]
    public async Task Aidn2_Machine_RejectedOnDifferentMachine()
    {
        await Task.Yield();
        var result = Validate(LicenseTestSupport.SignedKeyV2(mach: "not-this-machine-hash"));
        Assert.Equal(LicenseKeyStatus.Invalid, result.Status);
        Assert.Contains("machine", result.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Timeout = 60000)]
    public async Task Aidn2_Machine_AcceptedOnMatchingMachine()
    {
        await Task.Yield();
        var result = Validate(LicenseTestSupport.SignedKeyV2(mach: LicenseValidator.GetMachineIdHash()));
        Assert.Equal(LicenseKeyStatus.Active, result.Status);
    }

    // ───────────── CRL revocation ─────────────

    [Fact(Timeout = 60000)]
    public async Task Aidn2_Revocation_RevokedJtiIsRejected()
    {
        await Task.Yield();
        try
        {
            LicenseRevocationProvider.OverrideForTesting(
                LicenseTestSupport.SignedCrlV2(revokedJti: new[] { "leaked-token-1" }), DateTimeOffset.UtcNow);

            var result = Validate(LicenseTestSupport.SignedKeyV2(jti: "leaked-token-1"));
            Assert.Equal(LicenseKeyStatus.Revoked, result.Status);
        }
        finally
        {
            LicenseRevocationProvider.OverrideForTesting(null, DateTimeOffset.UtcNow);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Aidn2_Revocation_NonRevokedJtiIsActive()
    {
        await Task.Yield();
        try
        {
            LicenseRevocationProvider.OverrideForTesting(
                LicenseTestSupport.SignedCrlV2(revokedJti: new[] { "some-other-token" }), DateTimeOffset.UtcNow);

            var result = Validate(LicenseTestSupport.SignedKeyV2(jti: "my-token"));
            Assert.Equal(LicenseKeyStatus.Active, result.Status);
        }
        finally
        {
            LicenseRevocationProvider.OverrideForTesting(null, DateTimeOffset.UtcNow);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Aidn2_Revocation_ExpiredCrlIsIgnored_TokenStaysActive()
    {
        await Task.Yield();
        try
        {
            // A CRL that expired yesterday is stale → not enforced (fail-open); the token stays Active.
            LicenseRevocationProvider.OverrideForTesting(
                LicenseTestSupport.SignedCrlV2(revokedJti: new[] { "leaked-token-1" },
                    exp: DateTimeOffset.UtcNow.AddDays(-1)), DateTimeOffset.UtcNow);

            var result = Validate(LicenseTestSupport.SignedKeyV2(jti: "leaked-token-1"));
            Assert.Equal(LicenseKeyStatus.Active, result.Status);
        }
        finally
        {
            LicenseRevocationProvider.OverrideForTesting(null, DateTimeOffset.UtcNow);
        }
    }

    private sealed class ScopeEnv : IDisposable
    {
        private readonly string? _previous;
        public ScopeEnv(string? value)
        {
            _previous = Environment.GetEnvironmentVariable("AIDOTNET_LICENSE_SCOPE");
            Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_SCOPE", value);
        }
        public void Dispose() => Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_SCOPE", _previous);
    }
}
