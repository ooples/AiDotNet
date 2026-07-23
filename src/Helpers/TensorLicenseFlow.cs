using System.Reflection;
using AiDotNet.Models;

namespace AiDotNet.Helpers;

/// <summary>
/// Forward-compatible bridge between the upstream AiDotNet
/// <see cref="ModelPersistenceGuard"/> and the tensor-layer
/// <c>AiDotNet.Tensors.Licensing.PersistenceGuard</c>.
/// </summary>
/// <remarks>
/// <para>
/// Issue #1195 wires the upstream license key into the tensor-layer guard
/// (so the tensor reader/writer enforcement sees the same key) and wraps
/// upstream save/load calls that internally invoke a tensor reader in
/// <c>TensorsGuard.InternalOperation()</c> (so the trial counter ticks
/// once per user-facing op, not twice).
/// </para>
/// <para>
/// The issue's coordinated rollout ships the AiDotNet.Tensors release
/// with the new <c>AiDotNet.Tensors.Licensing</c> namespace
/// <em>after</em> this upstream release. The currently-published
/// <c>AiDotNet.Tensors</c> NuGet (0.57.0 at time of writing) does not
/// expose those types, so a direct <c>using</c> reference would
/// compile-fail.
/// </para>
/// <para>
/// This class therefore looks up the tensor-side guard by reflection at
/// runtime. When the Tensors package is bumped to a version that ships
/// the API, the bridge lights up automatically with no code change here.
/// While the API is absent, the bridge degrades to no-op — exactly the
/// state the issue describes ("Trial-fallback users are unaffected;
/// real keys validate as Active with an empty capability set on the
/// tensor side"). Once Tensors ships, capability-scoped enforcement
/// activates without a follow-up upstream PR.
/// </para>
/// </remarks>
internal static class TensorLicenseFlow
{
    private const string TensorsAssemblyName = "AiDotNet.Tensors";
    private const string TensorsLicenseKeyTypeName = "AiDotNet.Tensors.Licensing.AiDotNetTensorsLicenseKey";
    private const string TensorsGuardTypeName = "AiDotNet.Tensors.Licensing.PersistenceGuard";

    private static readonly object _resolutionLock = new();
    // Volatile guards the double-checked-lock pattern: a non-volatile read
    // of _resolved outside the lock could see a stale `false` (forcing a
    // redundant lock acquisition — harmless) OR see `true` while the
    // dependent reflection-handle writes (_setActiveLicenseKeyMethod et al.)
    // are still in CPU store buffers and not yet visible to other cores.
    // Volatile reads emit the appropriate acquire fence so once we observe
    // _resolved == true, all writes that preceded the corresponding
    // _resolved = true store are visible.
    private static volatile bool _resolved;

    // Cached reflection handles. All null when the Tensors licensing API
    // isn't present in the currently-loaded Tensors assembly (the no-op
    // state). Resolved once on first use.
    private static ConstructorInfo? _licenseKeyCtor;
    private static PropertyInfo? _serverUrlProp;
    private static PropertyInfo? _environmentProp;
    private static PropertyInfo? _gracePeriodProp;
    private static PropertyInfo? _telemetryProp;
    private static MethodInfo? _setActiveLicenseKeyMethod;
    private static MethodInfo? _internalOperationMethod;

    /// <summary>
    /// Sets the tensor-layer's active license key for the current logical
    /// call context. Returns an <see cref="IDisposable"/> that restores the
    /// previous tensor-layer key on dispose. When the tensor-layer
    /// licensing API is not present in the loaded
    /// <c>AiDotNet.Tensors</c> assembly, returns a no-op disposable.
    /// </summary>
    /// <remarks>
    /// Implements issue #1195 §2b. Both layers' <c>AsyncLocal</c>-backed
    /// scopes flow with the call context, so the inner build code
    /// transparently sees both keys.
    /// </remarks>
    public static IDisposable SetActiveLicenseKey(AiDotNetLicenseKey? key)
    {
        if (key is null) return NoopScope.Instance;
        EnsureResolved();
        if (_setActiveLicenseKeyMethod is null) return NoopScope.Instance;

        object? tensorsKey = ToTensorsKey(key);
        if (tensorsKey is null) return NoopScope.Instance;

        try
        {
            // Method signature on the tensor side: SetActiveLicenseKey(key) → IDisposable
            object? scope = _setActiveLicenseKeyMethod.Invoke(null, new[] { tensorsKey });
            return scope as IDisposable ?? NoopScope.Instance;
        }
        catch (Exception ex) when (
            ex is TargetInvocationException
                or ArgumentException
                or MethodAccessException
                or InvalidOperationException
                or TargetException
                or TargetParameterCountException)
        {
            // Same broad catch-and-degrade rationale as in ToTensorsKey:
            // the tensor-layer API may change signatures in a future
            // release in ways that produce these reflection failures
            // outside TargetInvocationException. The bridge must never
            // fault the upstream persistence path.
            System.Diagnostics.Trace.TraceWarning(
                $"TensorLicenseFlow: failed to set tensor-layer license key: {DescribeExceptionType(ex)}");
            return NoopScope.Instance;
        }
    }

    /// <summary>
    /// Marks the current logical call context as executing inside an upstream
    /// persistence operation that will internally call a tensor reader/writer.
    /// While the returned scope is active, the tensor-layer guard suppresses
    /// its own enforcement to avoid double-counting against the trial limit.
    /// When the tensor-layer licensing API is not present, returns a no-op
    /// disposable.
    /// </summary>
    /// <remarks>
    /// Implements issue #1195 §2c. The matching upstream
    /// <see cref="ModelPersistenceGuard.InternalOperation"/> already
    /// suppresses upstream enforcement; pairing the two means a single
    /// user-facing save/load invokes exactly one trial-counter tick.
    /// </remarks>
    public static IDisposable InternalOperation()
    {
        EnsureResolved();
        if (_internalOperationMethod is null) return NoopScope.Instance;

        try
        {
            object? scope = _internalOperationMethod.Invoke(null, parameters: null);
            return scope as IDisposable ?? NoopScope.Instance;
        }
        catch (Exception ex) when (
            ex is TargetInvocationException
                or ArgumentException
                or MethodAccessException
                or InvalidOperationException
                or TargetException
                or TargetParameterCountException)
        {
            System.Diagnostics.Trace.TraceWarning(
                $"TensorLicenseFlow: failed to enter tensor-layer InternalOperation: {DescribeExceptionType(ex)}");
            return NoopScope.Instance;
        }
    }

    /// <summary>
    /// True when the tensor-layer licensing API is present in the loaded
    /// <c>AiDotNet.Tensors</c> assembly. Exposed for tests so a regression
    /// test can assert the wrapper *would* fire when the API is available
    /// without depending on a specific Tensors version being installed.
    /// </summary>
    internal static bool IsTensorsLicensingAvailable
    {
        get
        {
            EnsureResolved();
            return _setActiveLicenseKeyMethod is not null
                && _internalOperationMethod is not null
                && _licenseKeyCtor is not null;
        }
    }

    private static void EnsureResolved()
    {
        if (_resolved) return;
        lock (_resolutionLock)
        {
            if (_resolved) return;

            try
            {
                Type? guardType = Type.GetType($"{TensorsGuardTypeName}, {TensorsAssemblyName}", throwOnError: false);
                Type? licenseKeyType = Type.GetType($"{TensorsLicenseKeyTypeName}, {TensorsAssemblyName}", throwOnError: false);

                if (guardType is null || licenseKeyType is null)
                {
                    // Tensors API is not present in the loaded assembly —
                    // this is the expected state for AiDotNet.Tensors
                    // versions prior to the issue #1195 follow-up release.
                    return;
                }

                _licenseKeyCtor = licenseKeyType.GetConstructor(new[] { typeof(string) });
                _serverUrlProp = licenseKeyType.GetProperty("ServerUrl");
                _environmentProp = licenseKeyType.GetProperty("Environment");
                _gracePeriodProp = licenseKeyType.GetProperty("OfflineGracePeriod");
                _telemetryProp = licenseKeyType.GetProperty("EnableTelemetry");

                _setActiveLicenseKeyMethod = guardType.GetMethod(
                    "SetActiveLicenseKey",
                    BindingFlags.Public | BindingFlags.Static,
                    binder: null,
                    types: new[] { licenseKeyType },
                    modifiers: null);
                _internalOperationMethod = guardType.GetMethod(
                    "InternalOperation",
                    BindingFlags.Public | BindingFlags.Static,
                    binder: null,
                    types: System.Type.EmptyTypes,
                    modifiers: null);
            }
            catch (Exception ex)
            {
                // Reflection probing must never fault the upstream
                // persistence path. Log and degrade to no-op.
                System.Diagnostics.Trace.TraceWarning(
                    $"TensorLicenseFlow: reflection probing failed: {DescribeExceptionType(ex)}");
            }
            finally
            {
                _resolved = true;
            }
        }
    }

    /// <summary>
    /// Builds an <c>AiDotNetTensorsLicenseKey</c> from an upstream
    /// <see cref="AiDotNetLicenseKey"/>. Field-by-field copy of the key
    /// string and the four optional configuration knobs the tensor side
    /// exposes. Issue #1195 §2b.
    /// </summary>
    private static object? ToTensorsKey(AiDotNetLicenseKey key)
    {
        if (_licenseKeyCtor is null) return null;

        try
        {
            object instance = _licenseKeyCtor.Invoke(new object[] { key.Key });

            if (key.ServerUrl is not null) _serverUrlProp?.SetValue(instance, key.ServerUrl);
            if (key.Environment is not null) _environmentProp?.SetValue(instance, key.Environment);
            _gracePeriodProp?.SetValue(instance, key.OfflineGracePeriod);
            _telemetryProp?.SetValue(instance, key.EnableTelemetry);

            return instance;
        }
        catch (Exception ex) when (
            ex is TargetInvocationException
                or ArgumentException
                or MethodAccessException
                or InvalidOperationException
                or TargetException
                or TargetParameterCountException)
        {
            // Reflection construction can throw beyond TargetInvocationException
            // when the future tensor-layer API changes signatures (e.g., a
            // setter renamed, a property type changed, a constructor parameter
            // added). The bridge must never fault the upstream persistence
            // path: log and degrade to no-op so the upstream guard still
            // runs. Includes the inner exception type for triage.
            System.Diagnostics.Trace.TraceWarning(
                $"TensorLicenseFlow: failed to build tensor-layer license key: {DescribeExceptionType(ex)}");
            return null;
        }
    }

    /// <summary>
    /// Builds a redaction-safe diagnostic string for an exception:
    /// the outer type name and (when wrapped in
    /// <see cref="TargetInvocationException"/>) the inner type name.
    /// Deliberately omits <c>ex.Message</c> and stack traces — these
    /// can include reflection-surfaced license-key content or other
    /// sensitive config in the licensing path. Type names alone are
    /// enough to triage the failure category without leaking secrets
    /// to operator-readable trace logs.
    /// </summary>
    private static string DescribeExceptionType(Exception ex)
    {
        if ((ex as TargetInvocationException)?.InnerException is { } inner)
        {
            return $"{ex.GetType().Name} ({inner.GetType().Name})";
        }
        return ex.GetType().Name;
    }

    private sealed class NoopScope : IDisposable
    {
        public static readonly NoopScope Instance = new();
        public void Dispose() { }
    }
}
