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
    /// Ambient nesting counter for internal persistence operations.
    /// When greater than zero, <see cref="EnforceBeforeSerialize"/> and
    /// <see cref="EnforceBeforeDeserialize"/> are no-ops to avoid double-counting.
    /// Uses a counter instead of a boolean to support nested InternalOperation scopes.
    /// </summary>
    /// <remarks>
    /// Uses <see cref="AsyncLocal{T}"/> instead of <c>[ThreadStatic]</c>: an
    /// <c>await</c> inside an <see cref="InternalOperation"/> scope can resume
    /// on a different thread-pool thread, at which point a thread-static
    /// counter reads as <c>0</c> on the new thread and the guard fires on an
    /// operation that should have been suppressed. <see cref="AsyncLocal{T}"/>
    /// flows with the logical call context, so the counter survives the
    /// continuation.
    /// </remarks>
    private static readonly AsyncLocal<int> _internalOperationDepth = new();

    /// <summary>
    /// Marks the current logical call context as executing an internal persistence operation.
    /// While the returned scope is active, <see cref="EnforceBeforeSerialize"/> and
    /// <see cref="EnforceBeforeDeserialize"/> will not count operations.
    /// Supports nesting — only the outermost scope's disposal re-enables enforcement.
    /// </summary>
    /// <remarks>
    /// Issue #1195 §2c: also enters the tensor-layer's
    /// <c>PersistenceGuard.InternalOperation()</c> when the tensor licensing
    /// API is present, so an upstream save/load that internally invokes a
    /// tensor reader/writer (e.g., <c>.gguf</c> / <c>.safetensors</c>) ticks
    /// the trial counter exactly once at the upstream layer rather than
    /// double-counting through the tensor layer too.
    /// </remarks>
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
        _internalOperationDepth.Value++;
        // Acquire the tensor-side scope FIRST so a partial failure (the
        // tensor scope acquisition throws) leaves the upstream depth
        // counter consistent — the local InternalOperationScope will still
        // decrement on dispose. Using a composite ensures the two scopes
        // are released in LIFO order.
        IDisposable tensorScope = TensorLicenseFlow.InternalOperation();
        return new InternalOperationScope(tensorScope);
    }

    /// <summary>
    /// Enforces trial/license requirements before a save operation.
    /// Must be called at the start of every SaveModel() implementation.
    /// </summary>
    /// <remarks>
    /// Save/Load are the user-facing persistence entry points and ALWAYS
    /// enforce — they are not suppressed by an outer
    /// <see cref="InternalOperation"/> scope. Only
    /// <see cref="EnforceBeforeSerialize"/> and
    /// <see cref="EnforceBeforeDeserialize"/> are suppressed when nested
    /// inside <see cref="InternalOperation"/> (those are the byte-level
    /// helpers that an already-guarded Save/Load may call into; suppressing
    /// them avoids double-counting). Suppressing Save/Load itself would
    /// bypass licensing entirely on every nested save chain.
    /// </remarks>
    /// <exception cref="LicenseRequiredException">
    /// Thrown when the free trial has expired and no valid license is configured.
    /// </exception>
    internal static void EnforceBeforeSave()
    {
        // Intentionally NOT short-circuiting on _internalOperationDepth.
        // See the remarks above — the scope is for byte-level Serialize/
        // Deserialize calls, not for user-facing Save/Load entry points.
        EnforceCore();
    }

    /// <summary>
    /// Enforces trial/license requirements before a load operation.
    /// Must be called at the start of every LoadModel()/Deserialize() file-based implementation.
    /// </summary>
    /// <remarks>
    /// Load is a user-facing entry point and is NOT suppressed by an outer
    /// <see cref="InternalOperation"/> scope. See
    /// <see cref="EnforceBeforeSave"/> for the rationale.
    /// </remarks>
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
        if (_internalOperationDepth.Value > 0) return;
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
        if (_internalOperationDepth.Value > 0) return;
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
    /// License key set by AiModelBuilder during BuildAsync. Flows through
    /// async continuations via <see cref="AsyncLocal{T}"/> so a
    /// <c>BuildAsync</c> that <c>await</c>s and resumes on a different
    /// thread still sees the same key when the guard fires on the post-
    /// await thread.
    /// </summary>
    private static readonly AsyncLocal<AiDotNet.Models.AiDotNetLicenseKey?> _activeBuilderLicenseKey = new();

    /// <summary>
    /// Trial-file path override used by the test suite so regression tests
    /// can exercise <see cref="EnforceCore"/> against an isolated trial.json
    /// without mutating <c>~/.aidotnet/trial.json</c> on the developer's
    /// real machine. <see langword="null"/> (default) means the real
    /// default path is used.
    /// </summary>
    /// <remarks>
    /// <see cref="AsyncLocal{T}"/>, not <c>[ThreadStatic]</c>: tests may
    /// <c>await</c> inside the scope (e.g., <c>AiModelBuilder.BuildAsync</c>
    /// regressions) and continuations often resume on a different thread-pool
    /// thread; a thread-static override would be lost after the await.
    /// </remarks>
    private static readonly AsyncLocal<string?> _testTrialFilePathOverride = new();

    /// <summary>
    /// Sets an isolated trial-file path for the current logical call context.
    /// Returns an <see cref="IDisposable"/> that restores the previous override
    /// on dispose. Intended for test isolation only — not a public API.
    /// </summary>
    /// <param name="path">The trial-file path to use for the duration of
    /// the scope. Pass <see langword="null"/> to clear any existing
    /// override.</param>
    internal static IDisposable SetTestTrialFilePathOverride(string? path)
    {
        string? previous = _testTrialFilePathOverride.Value;
        _testTrialFilePathOverride.Value = path;
        return new TestTrialFilePathOverrideScope(previous);
    }

    private sealed class TestTrialFilePathOverrideScope : IDisposable
    {
        private readonly string? _previous;
        public TestTrialFilePathOverrideScope(string? previous) => _previous = previous;
        public void Dispose() => _testTrialFilePathOverride.Value = _previous;
    }

    /// <summary>
    /// Sets the active builder license key for the current logical call context.
    /// Called by AiModelBuilder at the start of BuildAsync.
    /// </summary>
    /// <remarks>
    /// Issue #1195 §2b: also flows the key into the tensor layer when the
    /// tensor licensing API is present. Both layers' <c>AsyncLocal</c>-backed
    /// scopes flow with the call context, so inner build code transparently
    /// sees both keys and the disposal at the end of <c>BuildAsync</c>
    /// releases both.
    /// </remarks>
    internal static IDisposable SetActiveLicenseKey(AiDotNet.Models.AiDotNetLicenseKey? key)
    {
        AiDotNet.Models.AiDotNetLicenseKey? previous = _activeBuilderLicenseKey.Value;
        _activeBuilderLicenseKey.Value = key;
        IDisposable tensorScope = TensorLicenseFlow.SetActiveLicenseKey(key);
        return new ActiveLicenseKeyScope(previous, tensorScope);
    }

    private sealed class ActiveLicenseKeyScope : IDisposable
    {
        private readonly AiDotNet.Models.AiDotNetLicenseKey? _previous;
        private readonly IDisposable _tensorScope;
        public ActiveLicenseKeyScope(AiDotNet.Models.AiDotNetLicenseKey? previous, IDisposable tensorScope)
        {
            _previous = previous;
            _tensorScope = tensorScope;
        }
        public void Dispose()
        {
            // LIFO: release the tensor-side scope before clearing the
            // upstream value so any tensor-side teardown that consults the
            // upstream key still finds it.
            try { _tensorScope.Dispose(); }
            finally { _activeBuilderLicenseKey.Value = _previous; }
        }
    }

    /// <summary>
    /// Shared enforcement logic: checks for a license key first (with server validation),
    /// then falls back to trial operation counting. Emits anonymous telemetry events if enabled.
    /// </summary>
    private static void EnforceCore()
    {
        // Check if a license key is available:
        // 1. Builder's configured key (thread-static, set during BuildAsync)
        // 2. Environment variable (AIDOTNET_LICENSE_KEY)
        // 3. File (~/.aidotnet-license)
        string? resolvedKey = LicenseKeyResolver.Resolve(_activeBuilderLicenseKey.Value);
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
        string? overridePath = _testTrialFilePathOverride.Value;
        var trialManager = overridePath is not null
            ? new TrialStateManager(overridePath)
            : new TrialStateManager();
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
    /// Disposable scope that decrements the internal operation depth on dispose
    /// AND releases the tensor-layer InternalOperation scope acquired in
    /// <see cref="InternalOperation"/>. The two are paired so a single user-
    /// facing save/load yields exactly one trial-counter tick (issue #1195).
    /// </summary>
    private sealed class InternalOperationScope : IDisposable
    {
        private readonly IDisposable _tensorScope;
        public InternalOperationScope(IDisposable tensorScope) => _tensorScope = tensorScope;

        public void Dispose()
        {
            // Release tensor side first so any tensor-side teardown that
            // peeks at the upstream depth counter still sees the active
            // value. Then decrement the upstream counter.
            try
            {
                _tensorScope.Dispose();
            }
            finally
            {
                int current = _internalOperationDepth.Value;
                if (current > 0)
                {
                    _internalOperationDepth.Value = current - 1;
                }
            }
        }
    }
}
