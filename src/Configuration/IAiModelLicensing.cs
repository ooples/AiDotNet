using AiDotNet.Helpers;
using AiDotNet.Models;

namespace AiDotNet.Configuration;

/// <summary>
/// Component that owns the enterprise-license configuration for an AI model build. Extracted
/// from <c>AiModelBuilder</c> as slice 12 of the audit-2026-05 phase-2a DI refactor. Holds both
/// the user-supplied key and the cached <see cref="LicenseValidator"/> derived from it; the
/// component invalidates the validator any time the key changes so callers never see a stale
/// validation result.
/// </summary>
internal interface IAiModelLicensing
{
    /// <summary>The configured license key (from constructor or <see cref="ConfigureLicenseKey"/>), or <c>null</c> if not supplied.</summary>
    AiDotNetLicenseKey? LicenseKey { get; }

    /// <summary>Cached license validator. <c>null</c> until BuildAsync resolves it; the component clears this slot any time the key changes.</summary>
    LicenseValidator? Validator { get; set; }

    /// <summary>Sets the license key and clears the cached validator (subsequent BuildAsync calls re-resolve).</summary>
    void ConfigureLicenseKey(AiDotNetLicenseKey licenseKey);
}
