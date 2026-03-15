using AiDotNet.Enums;
using AiDotNet.Exceptions;
using AiDotNet.Models;

namespace AiDotNet.Helpers;

/// <summary>
/// Centralized enforcement point for model persistence licensing.
/// All SaveModel/LoadModel and Serialize/Deserialize implementations must call through
/// this guard to ensure trial/license compliance.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This class acts as a gatekeeper for all model save and load operations.
/// It checks whether you have a valid license or are within the free trial period before
/// allowing the operation to proceed. Training and inference are never restricted.</para>
/// <para>
/// Internal code paths (such as <c>ModelLoader.SaveEncrypted</c>) that need to call
/// <c>Serialize()</c> or <c>Deserialize()</c> as part of an already-guarded save/load
/// operation use <see cref="InternalOperation"/> to suppress double-counting.
/// </para>
/// </remarks>
internal static class ModelPersistenceGuard
{
    /// <summary>
    /// Thread-static nesting counter for internal persistence operations.
    /// When greater than zero, <see cref="EnforceBeforeSerialize"/> and
    /// <see cref="EnforceBeforeDeserialize"/> are no-ops to avoid double-counting.
    /// Uses a counter instead of a boolean to support nested InternalOperation scopes.
    /// </summary>
    [ThreadStatic]
    private static int _internalOperationDepth;

    /// <summary>
    /// Marks the current thread as executing an internal persistence operation.
    /// While the returned scope is active, <see cref="EnforceBeforeSerialize"/> and
    /// <see cref="EnforceBeforeDeserialize"/> will not count operations.
    /// Supports nesting — only the outermost scope's disposal re-enables enforcement.
    /// </summary>
    /// <returns>An <see cref="IDisposable"/> that decrements the depth counter when disposed.</returns>
    /// <example>
    /// <code>
    /// using (ModelPersistenceGuard.InternalOperation())
    /// {
    ///     byte[] bytes = model.Serialize(); // Will not trigger guard
    /// }
    /// </code>
    /// </example>
    internal static IDisposable InternalOperation()
    {
        _internalOperationDepth++;
        return new InternalOperationScope();
    }

    /// <summary>
    /// Enforces trial/license requirements before a save operation.
    /// Must be called at the start of every SaveModel() implementation.
    /// </summary>
    /// <exception cref="LicenseRequiredException">
    /// Thrown when the free trial has expired and no valid license is configured.
    /// </exception>
    internal static void EnforceBeforeSave()
    {
        EnforceCore();
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
        EnforceCore();
    }

    /// <summary>
    /// Enforces trial/license requirements before a Serialize() call.
    /// This is a no-op when called from an internal pipeline (e.g., ModelLoader)
    /// that already performed its own enforcement via <see cref="EnforceBeforeSave"/>.
    /// </summary>
    /// <exception cref="LicenseRequiredException">
    /// Thrown when called directly by user code and the free trial has expired.
    /// </exception>
    internal static void EnforceBeforeSerialize()
    {
        if (_internalOperationDepth > 0) return;
        EnforceCore();
    }

    /// <summary>
    /// Enforces trial/license requirements before a Deserialize() call.
    /// This is a no-op when called from an internal pipeline (e.g., ModelLoader)
    /// that already performed its own enforcement via <see cref="EnforceBeforeLoad"/>.
    /// </summary>
    /// <exception cref="LicenseRequiredException">
    /// Thrown when called directly by user code and the free trial has expired.
    /// </exception>
    internal static void EnforceBeforeDeserialize()
    {
        if (_internalOperationDepth > 0) return;
        EnforceCore();
    }

    /// <summary>
    /// Cached LicenseValidator instance. Created once when a license key is first resolved,
    /// then reused for subsequent validations (the validator caches its server response internally).
    /// </summary>
    private static LicenseValidator? _validator;
    private static string? _validatorKey;
    private static readonly object _validatorLock = new();

    /// <summary>
    /// Shared enforcement logic: checks for a license key first (with server validation),
    /// then falls back to trial operation counting. Emits anonymous telemetry events if enabled.
    /// </summary>
    private static void EnforceCore()
    {
        // Check if a license key is available (via env var or file)
        string? resolvedKey = LicenseKeyResolver.Resolve(null);
        if (resolvedKey is not null && resolvedKey.Trim().Length > 0)
        {
            string licenseKey = resolvedKey.Trim();
            var result = ValidateLicenseKey(licenseKey);

            switch (result.Status)
            {
                case LicenseKeyStatus.Active:
                case LicenseKeyStatus.ValidationPending:
                    // Active or pending (within grace period) — allow the operation
                    LicensingTelemetryCollector.Instance.RecordLicensedOperation("persistence");
                    return;

                case LicenseKeyStatus.Expired:
                    LicensingTelemetryCollector.Instance.RecordLicensingError("license_expired");
                    throw new LicenseRequiredException(
                        TrialExpirationReason.LicenseExpired);

                case LicenseKeyStatus.Revoked:
                    LicensingTelemetryCollector.Instance.RecordLicensingError("license_revoked");
                    throw new LicenseRequiredException(
                        TrialExpirationReason.LicenseInvalid);

                case LicenseKeyStatus.SeatLimitReached:
                    LicensingTelemetryCollector.Instance.RecordLicensingError("seat_limit_reached");
                    throw new LicenseRequiredException(
                        TrialExpirationReason.SeatLimitReached);

                case LicenseKeyStatus.Invalid:
                default:
                    LicensingTelemetryCollector.Instance.RecordLicensingError("license_invalid");
                    throw new LicenseRequiredException(
                        TrialExpirationReason.LicenseInvalid);
            }
        }

        // No license key — enforce trial limits
        var trialManager = new TrialStateManager();
        try
        {
            trialManager.RecordOperationOrThrow();

            // Record successful trial operation telemetry
            var status = trialManager.GetStatus();
            LicensingTelemetryCollector.Instance.RecordTrialOperation(
                status.OperationsUsed, status.OperationsRemaining, status.DaysElapsed);
        }
        catch (LicenseRequiredException ex)
        {
            // Record trial expiration telemetry
            LicensingTelemetryCollector.Instance.RecordTrialExpired(
                ex.ExpirationReason.ToString(),
                ex.OperationsPerformed ?? 0,
                ex.TrialDaysElapsed ?? 0);
            throw;
        }
    }

    /// <summary>
    /// Validates a license key using the cached <see cref="LicenseValidator"/> instance.
    /// Keys resolved from env vars or files are validated in offline-only mode (format/signature check only).
    /// Keys provided as explicit <see cref="AiDotNetLicenseKey"/> objects with a <c>ServerUrl</c>
    /// are validated against the server.
    /// </summary>
    /// <param name="licenseKey">The license key string to validate.</param>
    /// <param name="keyObject">The explicit license key object, or null if resolved from env/file.</param>
    private static LicenseValidationResult ValidateLicenseKey(string licenseKey, AiDotNetLicenseKey? keyObject = null)
    {
        LicenseValidator validator;

        lock (_validatorLock)
        {
            // Create a new validator if the key changed or no validator exists
            if (_validator is null || !string.Equals(_validatorKey, licenseKey, StringComparison.Ordinal))
            {
                AiDotNetLicenseKey keyConfig;
                if (keyObject is not null)
                {
                    // User-provided key object — respect their ServerUrl setting
                    keyConfig = keyObject;
                }
                else
                {
                    // Env var or file — offline-only validation (format/signature check, no server call)
                    keyConfig = new AiDotNetLicenseKey(licenseKey)
                    {
                        ServerUrl = string.Empty // explicit empty = offline-only mode
                    };
                }

                _validator = new LicenseValidator(keyConfig);
                _validatorKey = licenseKey;
            }

            validator = _validator;
        }

        return validator.Validate();
    }

    /// <summary>
    /// Disposable scope that resets the internal operation flag on disposal.
    /// </summary>
    private sealed class InternalOperationScope : IDisposable
    {
        public void Dispose()
        {
            if (_internalOperationDepth > 0)
            {
                _internalOperationDepth--;
            }
        }
    }
}
