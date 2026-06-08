using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Training;

/// <summary>
/// Reflection-based bridge to the AiDotNet.Tensors mixed-precision API
/// (<c>MixedPrecisionCompiledPlan</c>, <c>GradScaler</c>, <c>MixedPrecisionConfig</c>).
/// </summary>
/// <remarks>
/// <para>Why reflection? The mixed-precision types live on Tensors <c>main</c>
/// (introduced in <see href="https://github.com/ooples/AiDotNet.Tensors/pull/557">Tensors#557</see>)
/// but the latest NuGet release (<c>0.91.12</c>) predates that PR. AiDotNet's
/// FP16 activation-routing feature (<c>AIDOTNET_FP16_ACTIVATIONS</c>) needs
/// to compile against <c>0.91.12</c> without a hard reference to types that
/// don't exist in that assembly yet — otherwise the consumer-side build
/// breaks. Once Tensors publishes a release containing #557, the runtime
/// type-lookup succeeds and the FP16 path lights up automatically. No
/// AiDotNet code change required.</para>
/// <para>The reflection cost is paid once per shape (when a new compiled
/// plan is traced). The hot replay path goes through a cached
/// <see cref="MethodInfo"/> handle so per-step overhead is comparable to a
/// direct virtual call.</para>
/// </remarks>
internal static class MixedPrecisionReflection
{
    private static readonly object _lock = new();
    private static bool _probed;
    private static Type? _planType;
    private static Type? _scalerType;
    private static Type? _configType;
    private static MethodInfo? _traceMethod;
    private static MethodInfo? _stepMethod;
    private static MethodInfo? _stepAdamMethod;
    private static MethodInfo? _computeGradientsMethod;

    /// <summary>Returns true when the Tensors assembly in scope publishes the
    /// mixed-precision API surface (Tensors #557 or later). When this returns
    /// false, callers should fall back to the regular FP32 compiled-plan path.</summary>
    public static bool IsAvailable
    {
        get
        {
            if (_probed) return _planType is not null;
            lock (_lock)
            {
                if (_probed) return _planType is not null;
                ProbeUnlocked();
                _probed = true;
                return _planType is not null;
            }
        }
    }

    /// <summary>Returns true when the Tensors assembly in scope publishes the
    /// optimizer-agnostic <c>ComputeGradients</c> entry point (Tensors #574 or
    /// later). Strictly stronger than <see cref="IsAvailable"/>: a build can have
    /// the mixed-precision plan type (#557) without <c>ComputeGradients</c> (#574),
    /// in which case the generic fused-optimizer FP16 path must fall back to the
    /// regular FP32 compiled plan.</summary>
    public static bool IsComputeGradientsAvailable
    {
        get
        {
            // Touch IsAvailable to guarantee the one-time probe has run.
            return IsAvailable && _computeGradientsMethod is not null;
        }
    }

    private static void ProbeUnlocked()
    {
        // Walk loaded assemblies for the Tensors types. We don't hard-code the
        // assembly name (it could vary by build flavor / strong-name); the
        // namespace-qualified type name is unique enough.
        const string planFullName = "AiDotNet.Tensors.Engines.Compilation.MixedPrecisionCompiledPlan";
        const string scalerFullName = "AiDotNet.Tensors.Engines.Autodiff.GradScaler";
        const string configFullName = "AiDotNet.Tensors.Engines.Autodiff.MixedPrecisionConfig";

        foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
        {
            string? asmName = asm.GetName().Name;
            if (asmName is null || !asmName.StartsWith("AiDotNet.Tensors", StringComparison.Ordinal))
                continue;
            try
            {
                _planType ??= asm.GetType(planFullName, throwOnError: false);
                _scalerType ??= asm.GetType(scalerFullName, throwOnError: false);
                _configType ??= asm.GetType(configFullName, throwOnError: false);
            }
            catch
            {
                // Assembly can throw on GetType during type initialization
                // failures; skip it and continue probing.
            }
            if (_planType is not null && _scalerType is not null && _configType is not null) break;
        }

        Type? planType = _planType;
        if (planType is null) return;

        // Cache the method handles we'll dispatch through.
        _traceMethod = planType.GetMethod(
            "Trace",
            BindingFlags.Public | BindingFlags.Static,
            binder: null,
            types: new[] { typeof(Func<Tensor<float>>) },
            modifiers: null);

        _stepMethod = planType.GetMethod(
            "Step",
            BindingFlags.Public | BindingFlags.Instance,
            binder: null,
            types: new[] { typeof(IReadOnlyList<Tensor<float>>), typeof(float) },
            modifiers: null);

        // StepAdam's signature: StepAdam(IReadOnlyList<Tensor<float>>, double lr,
        //   double beta1, double beta2, double eps, double weightDecay, GradScaler).
        // We bind on parameter count + name match because we don't have the
        // GradScaler type expressed as a typeof() (it lives in the same probed
        // assembly). Walk the candidates and pick the public StepAdam with the
        // right shape.
        foreach (var m in planType.GetMethods(BindingFlags.Public | BindingFlags.Instance))
        {
            if (m.Name != "StepAdam") continue;
            var ps = m.GetParameters();
            if (ps.Length != 7) continue;
            if (ps[0].ParameterType != typeof(IReadOnlyList<Tensor<float>>)) continue;
            _stepAdamMethod = m;
            break;
        }

        // ComputeGradients(IReadOnlyList<Tensor<float>> parameters, GradScaler? scaler)
        // returns an optimizer-agnostic GradientResult (Loss/FoundInfNan/Gradients)
        // WITHOUT applying any update — Tensors #574, newer than the #557 plan type.
        // A Tensors build can therefore expose MixedPrecisionCompiledPlan (so
        // IsAvailable is true) yet predate ComputeGradients; gate the generic path
        // on IsComputeGradientsAvailable, not IsAvailable. Bind by name + the
        // 2-arg shape with the first parameter typed IReadOnlyList<Tensor<float>>
        // (the GradScaler param isn't expressible as a typeof here).
        foreach (var m in planType.GetMethods(BindingFlags.Public | BindingFlags.Instance))
        {
            if (m.Name != "ComputeGradients") continue;
            var ps = m.GetParameters();
            if (ps.Length != 2) continue;
            if (ps[0].ParameterType != typeof(IReadOnlyList<Tensor<float>>)) continue;
            _computeGradientsMethod = m;
            break;
        }
    }

    /// <summary>Trace a forward+loss thunk into a compiled mixed-precision plan.
    /// Returns the opaque plan handle (or null if the type isn't available).</summary>
    public static object? Trace(Func<Tensor<float>> forwardAndLoss)
    {
        if (!IsAvailable) return null;
        MethodInfo? trace = _traceMethod;
        if (trace is null) return null;
        return trace.Invoke(null, new object[] { forwardAndLoss });
    }

    /// <summary>Replay an SGD step against a previously-traced plan; returns the loss.</summary>
    public static float StepSgd(object plan, IReadOnlyList<Tensor<float>> parameters, float learningRate)
    {
        if (plan is null) throw new ArgumentNullException(nameof(plan));
        MethodInfo? step = _stepMethod;
        if (step is null) throw new InvalidOperationException("Mixed-precision API not available.");
        object? result = step.Invoke(plan, new object[] { parameters, learningRate });
        return ExtractLoss(result);
    }

    /// <summary>Construct a new <c>GradScaler</c> with the supplied loss scale.</summary>
    public static object? CreateGradScaler(float lossScale)
    {
        if (!IsAvailable) return null;
        Type? scalerType = _scalerType;
        Type? configType = _configType;
        if (scalerType is null || configType is null) return null;
        // MixedPrecisionConfig is a simple object with a settable LossScale property.
        object? cfg = Activator.CreateInstance(configType);
        if (cfg is null) return null;
        PropertyInfo? lsProp = configType.GetProperty("LossScale");
        if (lsProp is not null)
        {
            lsProp.SetValue(cfg, lossScale);
        }
        // GradScaler(MixedPrecisionConfig) constructor.
        return Activator.CreateInstance(scalerType, cfg);
    }

    /// <summary>Replay an Adam/AdamW step against a previously-traced plan. Returns the loss.</summary>
    public static float StepAdam(
        object plan, IReadOnlyList<Tensor<float>> parameters,
        double learningRate, double beta1, double beta2, double eps, double weightDecay,
        object gradScaler)
    {
        if (plan is null) throw new ArgumentNullException(nameof(plan));
        if (gradScaler is null) throw new ArgumentNullException(nameof(gradScaler));
        MethodInfo? stepAdam = _stepAdamMethod;
        if (stepAdam is null) throw new InvalidOperationException("Mixed-precision API not available.");
        object? result = stepAdam.Invoke(plan, new object[]
        {
            parameters, learningRate, beta1, beta2, eps, weightDecay, gradScaler
        });
        return ExtractLoss(result);
    }

    /// <summary>Run the optimizer-agnostic mixed-precision gradient pass against a
    /// previously-traced plan: forward + backward with FP16 activation storage,
    /// returning the unscaled FP32 per-parameter gradients WITHOUT applying any
    /// weight update. The returned <c>Gradients</c> list aligns 1:1 with
    /// <paramref name="parameters"/> (an entry may be null when the tape produced
    /// no gradient for that parameter). <c>FoundInfNan</c> is true when the scaler
    /// detected overflow — the caller must then skip the optimizer step.</summary>
    public static (float Loss, bool FoundInfNan, IReadOnlyList<Tensor<float>?> Gradients) ComputeGradients(
        object plan, IReadOnlyList<Tensor<float>> parameters, object gradScaler)
    {
        if (plan is null) throw new ArgumentNullException(nameof(plan));
        if (gradScaler is null) throw new ArgumentNullException(nameof(gradScaler));
        MethodInfo? compute = _computeGradientsMethod;
        if (compute is null)
            throw new InvalidOperationException("Mixed-precision ComputeGradients API not available.");
        object? result = compute.Invoke(plan, new object[] { parameters, gradScaler });
        return ExtractGradientResult(result);
    }

    /// <summary>Extract (Loss, FoundInfNan, Gradients) from the boxed <c>GradientResult</c>
    /// struct returned by <c>ComputeGradients</c>. The exact struct layout isn't known at
    /// compile time (the type lives in the probed Tensors assembly), so read each member by
    /// name, tolerating either a property or a field.</summary>
    private static (float Loss, bool FoundInfNan, IReadOnlyList<Tensor<float>?> Gradients) ExtractGradientResult(object? result)
    {
        if (result is null) return (0f, false, System.Array.Empty<Tensor<float>?>());
        Type t = result.GetType();

        // Accept float OR double Loss: the current API returns float, but a future
        // Tensors build could widen it to double — narrow rather than silently 0.
        float loss = 0f;
        object? lossVal = t.GetProperty("Loss")?.GetValue(result) ?? t.GetField("Loss")?.GetValue(result);
        if (lossVal is float lf) loss = lf;
        else if (lossVal is double ld) loss = (float)ld;

        bool foundInfNan = false;
        object? nanVal = t.GetProperty("FoundInfNan")?.GetValue(result) ?? t.GetField("FoundInfNan")?.GetValue(result);
        if (nanVal is bool nb) foundInfNan = nb;

        IReadOnlyList<Tensor<float>?> grads = System.Array.Empty<Tensor<float>?>();
        object? gradsVal = t.GetProperty("Gradients")?.GetValue(result) ?? t.GetField("Gradients")?.GetValue(result);
        if (gradsVal is IReadOnlyList<Tensor<float>?> typed)
        {
            grads = typed;
        }
        else if (gradsVal is System.Collections.IEnumerable seq)
        {
            // Defensive: if the published list element type differs (e.g. non-nullable
            // Tensor<float>), copy element-wise into the nullable-typed list we expose.
            var list = new List<Tensor<float>?>();
            foreach (object? item in seq) list.Add(item as Tensor<float>);
            grads = list;
        }

        return (loss, foundInfNan, grads);
    }

    /// <summary>Extract the loss from the boxed result of a Step / StepAdam invocation.
    /// The result is either a <c>float</c> directly or a result-struct exposing a
    /// <c>Loss</c> property/field; handle both shapes so we don't have to know the
    /// exact return-type layout at compile time.</summary>
    private static float ExtractLoss(object? result)
    {
        if (result is null) return 0f;
        if (result is float f) return f;
        if (result is double dd) return (float)dd;
        Type resultType = result.GetType();
        PropertyInfo? lossProp = resultType.GetProperty("Loss");
        if (lossProp is not null)
        {
            object? val = lossProp.GetValue(result);
            return AsFloat(val);
        }
        FieldInfo? lossField = resultType.GetField("Loss");
        if (lossField is not null)
        {
            object? val = lossField.GetValue(result);
            return AsFloat(val);
        }
        return 0f;
    }

    /// <summary>Narrow a boxed loss scalar to float, accepting either float or double
    /// (the API returns float today, but tolerate a future double widening).</summary>
    private static float AsFloat(object? val) => val switch
    {
        float fv => fv,
        double dv => (float)dv,
        _ => 0f,
    };
}
