using AiDotNet.Exceptions;

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
    /// Thread-static flag indicating that the current call is part of an internal
    /// persistence pipeline (e.g., ModelLoader encrypting/decrypting). When set,
    /// <see cref="EnforceBeforeSerialize"/> and <see cref="EnforceBeforeDeserialize"/>
    /// are no-ops to avoid double-counting operations.
    /// </summary>
    [ThreadStatic]
    private static bool _isInternalOperation;

    /// <summary>
    /// Marks the current thread as executing an internal persistence operation.
    /// While the returned scope is active, <see cref="EnforceBeforeSerialize"/> and
    /// <see cref="EnforceBeforeDeserialize"/> will not count operations.
    /// </summary>
    /// <returns>An <see cref="IDisposable"/> that resets the flag when disposed.</returns>
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
        _isInternalOperation = true;
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
        if (_isInternalOperation) return;
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
        if (_isInternalOperation) return;
        EnforceCore();
    }

    /// <summary>
    /// Shared enforcement logic: checks for a license key first, then falls back
    /// to trial operation counting. Emits anonymous telemetry events if enabled.
    /// </summary>
    private static void EnforceCore()
    {
        // If a license key is available (via env var or file), the user is licensed — allow
        string? licenseKey = LicenseKeyResolver.Resolve(null);
        if (!string.IsNullOrWhiteSpace(licenseKey))
        {
            LicensingTelemetryCollector.Instance.RecordLicensedOperation("persistence");
            return;
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
    /// Disposable scope that resets the internal operation flag on disposal.
    /// </summary>
    private sealed class InternalOperationScope : IDisposable
    {
        public void Dispose()
        {
            _isInternalOperation = false;
        }
    }
}
