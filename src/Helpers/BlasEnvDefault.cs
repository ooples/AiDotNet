using System;
#if NET6_0_OR_GREATER
using System.Runtime.CompilerServices;
#endif

namespace AiDotNet.Helpers;

/// <summary>
/// Default-enables the AiDotNet.Tensors BLAS fast-path before
/// <c>AiDotNet.Tensors.Helpers.BlasProvider</c>'s static initializer runs.
/// </summary>
/// <remarks>
/// <para>
/// AiDotNet.Tensors 0.75.3 (currently pinned in Directory.Packages.props)
/// ships a BlasProvider whose internal opt-in flag defaults to false —
/// i.e. libopenblas is NOT routed through unless the caller explicitly
/// sets <c>AIDOTNET_USE_BLAS=1</c> in the environment. The upstream
/// perf branch flips the default to "BLAS on when available" (matching
/// PyTorch / NumPy / TensorFlow), but that change isn't in 0.75.3 yet.
/// </para>
/// <para>
/// Without the BLAS fast-path engaged, every Tensor matmul / Conv2D
/// im2col+GEMM dispatch falls back to the in-house
/// <c>Im2ColHelper.MultiplyMatrixBlockedDouble</c> — which on paper-scale
/// ResNet50/VGG models leaves training at ~10 s/step locally and times out
/// at the test base's 120 s timeout on CI runners. Empirically, with BLAS
/// on the same ResNet50 step drops to ~8.5 s (a ~15% headroom that, paired
/// with the test base's existing 120 s budget, lets the 10-iteration paper-
/// scale training tests finish within timeout).
/// </para>
/// <para>
/// <b>How the gate works:</b> BlasProvider reads the env var exactly once
/// in its static field initializer. Once that initializer runs, env-var
/// changes have no effect. So we MUST set the env var BEFORE any code
/// touches BlasProvider — that's what <c>[ModuleInitializer]</c>
/// guarantees: it runs once, immediately after the AiDotNet assembly is
/// loaded, before any AiDotNet API the user might call (and certainly
/// before the first BlasProvider.IsAvailable probe is triggered
/// transitively by Tensor ops).
/// </para>
/// <para>
/// <b>Opt-out preserved:</b> if the user has already set
/// <c>AIDOTNET_USE_BLAS</c> (to any value — <c>1</c>, <c>0</c>,
/// <c>false</c>, etc.), we leave it alone. Only the unset / empty-string
/// case is overridden — exactly the "make the default sensible" hole the
/// upstream perf branch is fixing.
/// </para>
/// <para>
/// <b>net471:</b> [ModuleInitializer] is a .NET 5+ feature. The
/// initialization is skipped on net471. ResNet/VGG training timeouts
/// surface in net10.0 CI shards (08a, 08e) — net471 doesn't ship those
/// model-family invariants — so the gap doesn't matter for the
/// failing-test set this targets.
/// </para>
/// </remarks>
internal static class BlasEnvDefault
{
#if NET6_0_OR_GREATER
    // CA2255: ModuleInitializer is "intended for application code or
    // advanced source-generator scenarios". This IS the advanced
    // scenario the rule cites — we need BlasProvider's static init to
    // see the env var, and BlasProvider is in a NuGet package we can't
    // patch at use-site. The AiDotNet library is the choke-point for
    // every consumer, so library-level module-init is the right
    // attachment point.
#pragma warning disable CA2255
    [ModuleInitializer]
#pragma warning restore CA2255
    internal static void EnableBlasFastPathIfUnset()
    {
        // Documented escape hatch: hosted apps and CI shards that don't
        // want library-level env-var mutation can set the AppContext
        // switch "AiDotNet.DisableAutoBlasEnvDefault" to true. Honors the
        // pre-existing opt-out contract.
        if (AppContext.TryGetSwitch("AiDotNet.DisableAutoBlasEnvDefault", out var disabled) && disabled)
        {
            return;
        }

        var current = Environment.GetEnvironmentVariable("AIDOTNET_USE_BLAS");
        // Treat whitespace-only values the same as unset — a stray space
        // in a CI config (`AIDOTNET_USE_BLAS=" "`) should not silently
        // suppress the default-on behaviour.
        if (string.IsNullOrWhiteSpace(current))
        {
            // Industry-standard default. Mirrors PyTorch / NumPy / TF —
            // they all link against BLAS by default and don't require a
            // separate opt-in. The AiDotNet.Native.OpenBLAS NuGet is a
            // transitive dependency of every AiDotNet install so the
            // native lib is on disk; we just need the env switch.
            Environment.SetEnvironmentVariable("AIDOTNET_USE_BLAS", "1");
        }
    }
#endif
}
