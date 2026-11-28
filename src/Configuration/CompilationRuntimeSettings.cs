using System;
using System.Threading;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Optimization;

namespace AiDotNet.Configuration;

/// <summary>
/// Thread-safe process and thread-local bridge that applies <see cref="CompilationOptions"/> across
/// <c>TensorCodecOptions</c>, <c>DirectGpuTensorEngine</c>, <c>LayerBase&lt;T&gt;</c> derived-weight caches,
/// <c>GradientBasedOptimizerBase</c> GPU scalar buffers, <c>CompiledTapeTrainingStep</c>, and dynamic MoE routing.
/// </summary>
internal static class CompilationRuntimeSettings
{
    private static CompilationOptions s_default = CompilationOptions.Default;

    [ThreadStatic]
    private static CompilationOptions? t_current;

    internal static CompilationOptions Current => t_current ?? Volatile.Read(ref s_default);

    internal static bool IsCompilationEnabled => Current.Mode != ExecutionMode.Eager;

    internal static bool ThrowOnCompilationFailure => Current.Mode == ExecutionMode.StrictCompiled;

    internal static bool CudaGraphTreesEnabled
        => IsCompilationEnabled && Current.EnableCudaGraphTrees && AiDotNetEngine.Current.Unwrap() is DirectGpuTensorEngine;

    internal static bool MultiBucketShapeRoutingEnabled
        => IsCompilationEnabled && Current.EnableMultiBucketShapeRouting;

    internal static bool SegmentedBlockGraphsEnabled
        => IsCompilationEnabled && Current.EnableSegmentedBlockGraphs;

    internal static bool GpuResidentOptimizerScalarsEnabled
        => Current.EnableGpuResidentOptimizerScalars;

    internal static bool DevicePredicatedDynamicRoutingEnabled
        => Current.EnableDevicePredicatedDynamicRouting;

    internal static bool PreferDirectPtx
        => !Current.AllowExternalVendorLibraries || Current.PreferSelfContainedDirectPtx;

    /// <summary>
    /// Applies <paramref name="options"/> (plus any explicit overrides from <paramref name="jitConfig"/> when the
    /// caller also invoked <c>ConfigureJitCompilation</c>) to the active thread and engine runtime.
    /// </summary>
    internal static void Apply(
        CompilationOptions? options,
        JitCompilationConfig? jitConfig = null,
        bool setAsProcessDefault = false)
    {
        var resolved = options is not null ? new CompilationOptions(options) : CompilationOptions.Default;
        resolved.Validate();

        if (setAsProcessDefault)
        {
            Volatile.Write(ref s_default, resolved);
            DerivedTensorCacheSettings.DefaultEnabled = resolved.EnableDerivedWeightCaching;
        }

        t_current = resolved;
        DerivedTensorCacheSettings.SetThreadOverride(resolved.EnableDerivedWeightCaching);

        bool compileEnabled = resolved.Mode != ExecutionMode.Eager && (jitConfig?.Enabled ?? true);
        bool fusionEnabled = compileEnabled && resolved.EnableHybridFusionTileIr;
        bool mixedPrecision = resolved.PrecisionTier is ComputePrecisionTier.BF16 or ComputePrecisionTier.FP16
            || (jitConfig?.EnableMixedPrecisionCompilation ?? false);

        if (jitConfig is not null)
        {
            jitConfig.Validate();
        }

        var codecOptions = new TensorCodecOptions
        {
            EnableCompilation = compileEnabled,
            EnableDataflowFusion = fusionEnabled && (jitConfig?.EnableDataflowFusion ?? true),
            EnableAlgebraicBackward = fusionEnabled && (jitConfig?.EnableAlgebraicBackward ?? true),
            EnableSpectralDecomposition = jitConfig?.EnableSpectralDecomposition ?? false,
            SpectralErrorTolerance = jitConfig?.SpectralErrorTolerance ?? 1e-5f,
            DataflowFusionMaxHidden = jitConfig?.DataflowFusionMaxHidden ?? 512,
            EnableConvBnFusion = fusionEnabled && (jitConfig?.EnableConvBnFusion ?? true),
            EnableAttentionFusion = fusionEnabled && (jitConfig?.EnableAttentionFusion ?? true),
            EnablePointwiseFusion = fusionEnabled && (jitConfig?.EnablePointwiseFusion ?? true),
            EnableConstantFolding = compileEnabled && (jitConfig?.EnableConstantFolding ?? true),
            EnableForwardCSE = compileEnabled && (jitConfig?.EnableForwardCSE ?? true),
            EnableBlasBatch = compileEnabled && (jitConfig?.EnableBlasBatch ?? true),
            EnableMixedPrecision = mixedPrecision,
            UseCublas = resolved.AllowExternalVendorLibraries,
            UseCudnn = resolved.AllowExternalVendorLibraries,
            UseCudnnBatchNorm = resolved.AllowExternalVendorLibraries,
        };
        TensorCodecOptions.SetCurrent(codecOptions);

        TryConfigureTensorBackendRuntime(resolved, compileEnabled);
    }

    /// <summary>
    /// Converts <paramref name="options"/> into a compatible <see cref="JitCompilationConfig"/> view so legacy callers
    /// inspecting <c>JitCompilationConfig</c> observe consistent flags.
    /// </summary>
    internal static JitCompilationConfig ToJitCompilationConfig(CompilationOptions options, JitCompilationConfig? baseJit = null)
    {
        options.Validate();
        if (options.Mode == ExecutionMode.Eager)
        {
            return new JitCompilationConfig
            {
                Enabled = false,
                PersistCompiledProgram = options.PersistCompiledProgram,
                ThrowOnFailure = false,
            };
        }

        var source = baseJit ?? JitCompilationConfig.Default;
        return new JitCompilationConfig
        {
            Enabled = source.Enabled,
            PersistCompiledProgram = options.PersistCompiledProgram && source.PersistCompiledProgram,
            ThrowOnFailure = options.Mode == ExecutionMode.StrictCompiled || source.ThrowOnFailure,
            EnableDataflowFusion = options.EnableHybridFusionTileIr && source.EnableDataflowFusion,
            EnableAlgebraicBackward = options.EnableHybridFusionTileIr && source.EnableAlgebraicBackward,
            EnableSpectralDecomposition = source.EnableSpectralDecomposition,
            SpectralErrorTolerance = source.SpectralErrorTolerance,
            DataflowFusionMaxHidden = source.DataflowFusionMaxHidden,
            EnableConvBnFusion = options.EnableHybridFusionTileIr && source.EnableConvBnFusion,
            EnableAttentionFusion = options.EnableHybridFusionTileIr && source.EnableAttentionFusion,
            EnablePointwiseFusion = options.EnableHybridFusionTileIr && source.EnablePointwiseFusion,
            EnableConstantFolding = source.EnableConstantFolding,
            EnableForwardCSE = source.EnableForwardCSE,
            EnableBlasBatch = source.EnableBlasBatch,
            EnableMixedPrecisionCompilation =
                options.PrecisionTier is ComputePrecisionTier.BF16 or ComputePrecisionTier.FP16
                || source.EnableMixedPrecisionCompilation,
        };
    }

    private static void TryConfigureTensorBackendRuntime(CompilationOptions options, bool compileEnabled)
    {
        try
        {
            var tensorsAssembly = typeof(TensorCodecOptions).Assembly;
            bool allowTf32 = options.PrecisionTier == ComputePrecisionTier.TF32;
            var policyType = tensorsAssembly.GetType("AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaDispatchPolicy");
            if (policyType is not null)
            {
                TrySetStaticBoolProperty(policyType, "AllowTF32", allowTf32);
                TrySetStaticBoolProperty(policyType, "ConvReducedPrecisionTF32", allowTf32);
                TrySetStaticBoolProperty(policyType, "AllowVendorLibraries", options.AllowExternalVendorLibraries);
                TrySetStaticBoolProperty(policyType, "PreferDirectPtx", !options.AllowExternalVendorLibraries || options.PreferSelfContainedDirectPtx);
                TrySetStaticBoolProperty(policyType, "EnableTf32", allowTf32);
                TrySetStaticBoolProperty(policyType, "EnableMaxAutotune", options.EnableMaxAutotune);
                if (options.EnableMaxAutotune)
                {
                    var cacheProp = policyType.GetProperty("AutotuneCacheDirectory");
                    if (cacheProp is not null && cacheProp.CanWrite && cacheProp.PropertyType == typeof(string))
                    {
                        cacheProp.SetValue(null, options.ResolveEffectiveAutotuneCacheDirectory());
                    }
                }
            }

            // Enable static-shape inference CUDA Graph capture/replay on CompiledInferencePlan<T> when
            // CompilationOptions.EnableCudaGraphTrees is active, eliminating host-device step syncs during Predict().
            bool enableInferCudaGraph = compileEnabled && options.EnableCudaGraphTrees;
            var openPlanType = tensorsAssembly.GetType("AiDotNet.Tensors.Engines.Compilation.CompiledInferencePlan`1");
            if (openPlanType is not null)
            {
                SetInferenceCudaGraphOverride(openPlanType.MakeGenericType(typeof(float)), enableInferCudaGraph);
                SetInferenceCudaGraphOverride(openPlanType.MakeGenericType(typeof(double)), enableInferCudaGraph);
            }
        }
        catch (Exception)
        {
            // Optional cross-assembly runtime hints never fail model construction on older Tensors packages.
        }
    }

    private static void SetInferenceCudaGraphOverride(Type closedPlanType, bool enabled)
    {
        const System.Reflection.BindingFlags flags =
            System.Reflection.BindingFlags.Static |
            System.Reflection.BindingFlags.Public |
            System.Reflection.BindingFlags.NonPublic;
        var field = closedPlanType.GetField("InferenceCudaGraphOverride", flags);
        if (field is not null && field.FieldType == typeof(bool?))
        {
            field.SetValue(null, (bool?)enabled);
        }
    }

    private static void TrySetStaticBoolProperty(Type targetType, string propertyName, bool value)
    {
        var prop = targetType.GetProperty(propertyName);
        if (prop is not null && prop.CanWrite && prop.PropertyType == typeof(bool))
        {
            prop.SetValue(null, value);
        }
    }
}
