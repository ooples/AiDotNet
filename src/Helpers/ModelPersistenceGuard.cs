using AiDotNet.Exceptions;

namespace AiDotNet.Helpers;

/// <summary>
/// Centralized enforcement point for model persistence licensing.
/// All SaveModel/LoadModel implementations must call through this guard
/// to ensure trial/license compliance.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This class acts as a gatekeeper for all model save and load operations.
/// It checks whether you have a valid license or are within the free trial period before
/// allowing the operation to proceed. Training and inference are never restricted.</para>
/// </remarks>
internal static class ModelPersistenceGuard
{
    /// <summary>
    /// Enforces trial/license requirements before a save operation.
    /// Must be called at the start of every SaveModel() implementation.
    /// </summary>
    /// <exception cref="LicenseRequiredException">
    /// Thrown when the free trial has expired and no valid license is configured.
    /// </exception>
    internal static void EnforceBeforeSave()
    {
        // If a license key is available (via env var or file), the user is licensed — allow
        string? licenseKey = LicenseKeyResolver.Resolve(null);
        if (!string.IsNullOrWhiteSpace(licenseKey))
        {
            return;
        }

        // No license key — enforce trial limits
        var trialManager = new TrialStateManager();
        trialManager.RecordOperationOrThrow();
    }

    /// <summary>
    /// Enforces trial/license requirements before a load operation.
    /// Must be called at the start of every LoadModel()/Deserialize() file-based implementation.
    /// </summary>
    /// <exception cref="LicenseRequiredException">
    /// Thrown when the free trial has expired and no valid license is configured.
    /// </exception>
    internal static void EnforceBeforeLoad()
    {
        // If a license key is available (via env var or file), the user is licensed — allow
        string? licenseKey = LicenseKeyResolver.Resolve(null);
        if (!string.IsNullOrWhiteSpace(licenseKey))
        {
            return;
        }

        // No license key — enforce trial limits
        var trialManager = new TrialStateManager();
        trialManager.RecordOperationOrThrow();
    }
}
