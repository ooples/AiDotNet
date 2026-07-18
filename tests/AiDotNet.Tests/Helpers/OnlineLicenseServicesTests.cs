using System;
using System.IO;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.Helpers;

/// <summary>
/// Tests the v2 client glue: sibling-URL derivation, the cached offline aidn2 token fallback (used when the
/// server is unreachable), and the on-disk CRL cache round-trip. Runs in the License collection so the
/// fixture's test public key is embedded and offline aidn2 verification succeeds.
/// </summary>
[Collection("License")]
public class OnlineLicenseServicesTests
{
    private const string ServerKey = "AIDN-PROD-COMMUNITY-0123456789abcdef0123456789abcdef";

    // ───────────── sibling URL derivation ─────────────

    [Fact(Timeout = 60000)]
    public async Task DeriveFunctionUrl_SwapsTrailingSegment()
    {
        await Task.Yield();
        const string validate = "https://yfkqwpgjahoamlgckjib.supabase.co/functions/v1/validate-license";
        Assert.Equal("https://yfkqwpgjahoamlgckjib.supabase.co/functions/v1/get-revocations",
            OnlineLicenseServices.DeriveFunctionUrl(validate, "get-revocations"));
        Assert.Equal("https://yfkqwpgjahoamlgckjib.supabase.co/functions/v1/issue-license",
            OnlineLicenseServices.DeriveFunctionUrl(validate, "issue-license"));
    }

    [Fact(Timeout = 60000)]
    public async Task DeriveFunctionUrl_ReturnsNullForNonMatchingUrl()
    {
        await Task.Yield();
        Assert.Null(OnlineLicenseServices.DeriveFunctionUrl("https://example.test/custom-endpoint", "get-revocations"));
        Assert.Null(OnlineLicenseServices.DeriveFunctionUrl("", "get-revocations"));
    }

    // ───────────── cached offline token fallback ─────────────

    [Fact(Timeout = 60000)]
    public async Task CachedOfflineToken_ValidAndMachineBound_ReturnsActiveWithCaps()
    {
        await Task.Yield();
        using var dir = new TempCacheDir();
        using var _ = OnlineLicenseServices.OverrideCacheDirForTesting(dir.Path);

        string token = LicenseTestSupport.SignedKeyV2(
            mach: LicenseValidator.GetMachineIdHash(),
            caps: new[] { "model:save", "tensors:save" });
        OnlineLicenseServices.CacheOfflineTokenForTesting(ServerKey, token);

        var result = OnlineLicenseServices.TryValidateCachedOfflineToken(ServerKey);

        Assert.NotNull(result);
        Assert.Equal(LicenseKeyStatus.Active, result!.Status);
        Assert.True(result.HasCapability("model:save"));
        Assert.True(result.HasCapability("tensors:save"));
    }

    [Fact(Timeout = 60000)]
    public async Task CachedOfflineToken_BoundToDifferentMachine_ReturnsNull()
    {
        await Task.Yield();
        using var dir = new TempCacheDir();
        using var _ = OnlineLicenseServices.OverrideCacheDirForTesting(dir.Path);

        string token = LicenseTestSupport.SignedKeyV2(mach: "some-other-machine-hash");
        OnlineLicenseServices.CacheOfflineTokenForTesting(ServerKey, token);

        Assert.Null(OnlineLicenseServices.TryValidateCachedOfflineToken(ServerKey));
    }

    [Fact(Timeout = 60000)]
    public async Task CachedOfflineToken_Expired_ReturnsNull()
    {
        await Task.Yield();
        using var dir = new TempCacheDir();
        using var _ = OnlineLicenseServices.OverrideCacheDirForTesting(dir.Path);

        string token = LicenseTestSupport.SignedKeyV2(
            mach: LicenseValidator.GetMachineIdHash(),
            exp: DateTimeOffset.UtcNow.AddDays(-1));
        OnlineLicenseServices.CacheOfflineTokenForTesting(ServerKey, token);

        Assert.Null(OnlineLicenseServices.TryValidateCachedOfflineToken(ServerKey));
    }

    [Fact(Timeout = 60000)]
    public async Task CachedOfflineToken_None_ReturnsNull()
    {
        await Task.Yield();
        using var dir = new TempCacheDir();
        using var _ = OnlineLicenseServices.OverrideCacheDirForTesting(dir.Path);

        Assert.Null(OnlineLicenseServices.TryValidateCachedOfflineToken(ServerKey));
    }

    // ───────────── end-to-end: server unreachable → offline token grants Active ─────────────

    [Fact(Timeout = 60000)]
    public async Task Validate_ServerUnreachable_FallsBackToCachedOfflineToken()
    {
        await Task.Yield();
        using var dir = new TempCacheDir();
        using var _ = OnlineLicenseServices.OverrideCacheDirForTesting(dir.Path);

        // Cache a machine-bound offline token for this server key.
        string token = LicenseTestSupport.SignedKeyV2(
            mach: LicenseValidator.GetMachineIdHash(),
            caps: new[] { "model:save" });
        OnlineLicenseServices.CacheOfflineTokenForTesting(ServerKey, token);

        // Point at a port that refuses instantly so the online attempt fails fast (no 15s wait), driving the
        // catch → offline-token fallback path.
        var key = new AiDotNetLicenseKey(ServerKey) { ServerUrl = "http://127.0.0.1:1/validate-license" };
        var result = new LicenseValidator(key).Validate();

        Assert.Equal(LicenseKeyStatus.Active, result.Status);
        Assert.True(result.HasCapability("model:save"));
    }

    // ───────────── CRL disk cache round-trip ─────────────

    [Fact(Timeout = 60000)]
    public async Task ReadCachedCrl_RoundTripsWhatWasCached()
    {
        await Task.Yield();
        using var dir = new TempCacheDir();
        using var _ = OnlineLicenseServices.OverrideCacheDirForTesting(dir.Path);

        Assert.Null(OnlineLicenseServices.ReadCachedCrl());

        string crl = LicenseTestSupport.SignedCrlV2(revokedJti: new[] { "abc" });
        OnlineLicenseServices.CacheCrlForTesting(crl);

        Assert.Equal(crl, OnlineLicenseServices.ReadCachedCrl());
    }

    private sealed class TempCacheDir : IDisposable
    {
        public string Path { get; }

        public TempCacheDir()
        {
            Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "aidn-cache-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Path);
        }

        public void Dispose()
        {
            try { Directory.Delete(Path, recursive: true); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
    }
}
