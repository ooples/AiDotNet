using AiDotNet.Helpers;
using AiDotNet.Models;

namespace AiDotNet.Configuration;

/// <summary>
/// Default implementation of <see cref="IAiModelLicensing"/>. Audit-2026-05 phase-2a slice 12.
/// Mirrors the pre-refactor inline storage in <c>AiModelBuilder</c>: setting a new license key
/// always nulls the cached validator so the next BuildAsync re-resolves against the fresh key.
/// </summary>
internal class AiModelLicensing : IAiModelLicensing
{
    /// <summary>Sets the license key from the constructor. The component starts with this value but the caller may override via <see cref="ConfigureLicenseKey"/>.</summary>
    public AiModelLicensing(AiDotNetLicenseKey? initialLicenseKey = null)
    {
        LicenseKey = initialLicenseKey;
    }

    /// <inheritdoc/>
    public AiDotNetLicenseKey? LicenseKey { get; private set; }

    /// <inheritdoc/>
    public LicenseValidator? Validator { get; set; }

    /// <inheritdoc/>
    public void ConfigureLicenseKey(AiDotNetLicenseKey licenseKey)
    {
        Guard.NotNull(licenseKey);
        LicenseKey = licenseKey;
        Validator = null; // reset cached validator — next BuildAsync re-resolves against the new key
    }
}
