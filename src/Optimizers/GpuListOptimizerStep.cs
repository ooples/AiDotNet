using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Optimizers;

/// <summary>
/// Shared whole-parameter-list ("foreach") GPU step for gradient-based and streaming optimizers.
/// </summary>
/// <typeparam name="T">The optimizer's numeric type; GPU list steps execute for <see cref="float"/>.</typeparam>
internal static class GpuListOptimizerStep<T>
{
    /// <summary>
    /// Returns the optimizer state tensors one parameter's GPU list step reads and writes (in the order the list step
    /// takes them), creating them on first use, or null to leave that parameter to the fallback loop.
    /// </summary>
    internal delegate Tensor<T>[]? StateResolver(Tensor<T> param);

    /// <summary>
    /// Returns <paramref name="param"/>'s state in <paramref name="states"/>, creating it with
    /// <paramref name="create"/> on first use, matching each optimizer's per-parameter loop.
    /// </summary>
    internal static Tensor<T> GetOrAddState(
        IDictionary<Tensor<T>, Tensor<T>> states, Tensor<T> param, Func<Tensor<T>, Tensor<T>> create)
    {
        if (!states.TryGetValue(param, out var state))
        {
            state = create(param);
            states[param] = state;
        }
        return state;
    }

    /// <summary>
    /// Runs one optimizer step for every GPU-eligible parameter of <paramref name="context"/> as a single
    /// whole-parameter-list ("foreach") GPU step, <c>GpuOptimizer.&lt;listStepName&gt;</c>, instead of one GPU
    /// operation per parameter. Returns the parameters updated, or null when nothing was updated.
    /// </summary>
    internal static HashSet<Tensor<T>>? TryApply(
        TapeStepContext<T> context,
        IEngine engine,
        string listStepName,
        int stateCount,
        StateResolver resolveStates,
        params object[] scalarArguments)
    {
        if (typeof(T) != typeof(float) || context.Parameters.Count == 0) return null;
        var method = GpuListStepBridge.Resolve(listStepName, stateCount, scalarArguments);
        if (method is null) return null;

        int capacity = context.Parameters.Count;
        var parameters = new List<Tensor<float>>(capacity);
        var gradients = new List<Tensor<float>>(capacity);
        var states = new List<Tensor<float>>[stateCount];
        for (int s = 0; s < stateCount; s++) states[s] = new List<Tensor<float>>(capacity);
        var updated = new HashSet<Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);

        foreach (var param in context.Parameters)
        {
            if (SparseEmbeddingOptimizerHelpers.HasUnprocessedSparseEmbeddingGrad(context, param)) continue;
            if (!SparseEmbeddingOptimizerHelpers.TryGetEffectiveGradient(context, param, engine, out var grad)) continue;
            if (param.Length == 0 || param.Length != grad.Length) continue;
            var paramStates = resolveStates(param);
            if (paramStates is null || paramStates.Length != stateCount) continue;

            bool statesMatch = true;
            for (int s = 0; s < stateCount; s++)
            {
                if (paramStates[s] is not { } state || state.Length != param.Length)
                {
                    statesMatch = false;
                    break;
                }
            }
            if (!statesMatch || !updated.Add(param)) continue;

            parameters.Add((Tensor<float>)(object)param);
            gradients.Add((Tensor<float>)(object)grad);
            for (int s = 0; s < stateCount; s++) states[s].Add((Tensor<float>)(object)paramStates[s]);
        }

        if (parameters.Count == 0) return null;

        return GpuListStepBridge.TryInvokeLists(method, parameters, gradients, states, scalarArguments) ? updated : null;
    }
}

/// <summary>
/// Binds <c>GpuOptimizer</c>'s whole-parameter-list optimizer steps by reflection so <c>AiDotNet</c> compiles and
/// executes cleanly against <c>AiDotNet.Tensors</c> packages with or without each list method.
/// </summary>
internal static class GpuListStepBridge
{
    private static readonly ConcurrentDictionary<string, MethodInfo?> Methods = new();

    internal static MethodInfo? Resolve(string name, int stateCount, object[] scalarArguments)
    {
        var types = new Type[2 + stateCount + scalarArguments.Length];
        for (int i = 0; i < 2 + stateCount; i++) types[i] = typeof(IReadOnlyList<Tensor<float>>);
        for (int i = 0; i < scalarArguments.Length; i++) types[2 + stateCount + i] = scalarArguments[i].GetType();
        string key = name + "(" + string.Join(",", types.Select(t => t.Name)) + ")";
        return Methods.GetOrAdd(key, _ => Bind(name, types));
    }

    internal static bool TryInvokeLists(
        MethodInfo method,
        IReadOnlyList<Tensor<float>> parameters,
        IReadOnlyList<Tensor<float>> gradients,
        IReadOnlyList<Tensor<float>>[] states,
        object[] scalarArguments)
    {
        var arguments = new object[2 + states.Length + scalarArguments.Length];
        arguments[0] = parameters;
        arguments[1] = gradients;
        for (int s = 0; s < states.Length; s++) arguments[2 + s] = states[s];
        Array.Copy(scalarArguments, 0, arguments, 2 + states.Length, scalarArguments.Length);
        return Invoke(method, arguments);
    }

    internal static bool Invoke(MethodInfo method, object[] arguments)
    {
        try
        {
            return method.Invoke(null, arguments) is true;
        }
        catch (TargetInvocationException ex) when (ex.InnerException is { } inner)
        {
            ExceptionDispatchInfo.Capture(inner).Throw();
            throw;
        }
    }

    private static MethodInfo? Bind(string name, Type[] types)
    {
        try
        {
            var method = typeof(GpuOptimizer).GetMethod(
                name, BindingFlags.Public | BindingFlags.Static, binder: null, types: types, modifiers: null);
            return method is not null && method.ReturnType == typeof(bool) ? method : null;
        }
        catch (AmbiguousMatchException ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                $"[AiDotNet] GpuListStepBridge could not bind {name}: {ex.GetType().Name}.");
            return null;
        }
    }
}

/// <summary>
/// Step-scoped GPU batching for <see cref="GradientBasedOptimizerBase{T, TInput, TOutput}.Step(TapeStepContext{T})"/>:
/// holds a single <see cref="DirectGpuTensorEngine.EnterGpuOp"/> scope across the entire optimizer step and commits all
/// per-tensor GPU write sync points at the end of the step using one shared <see cref="GpuSyncPoint"/> lease group
/// rather than creating and recording an independent CUDA event per parameter and moment tensor.
/// </summary>
internal static class GradientBasedOptimizerGpuStepBatch
{
    internal readonly struct PendingCommit
    {
        public readonly Tensor<float> Tensor;
        public readonly IGpuBuffer? ParameterBuffer;
        public readonly bool IsParameter;

        public PendingCommit(Tensor<float> tensor, IGpuBuffer? parameterBuffer, bool isParameter)
        {
            Tensor = tensor;
            ParameterBuffer = parameterBuffer;
            IsParameter = isParameter;
        }
    }

    internal sealed class ActiveBatch
    {
        public readonly DirectGpuTensorEngine Engine;
        public readonly IDirectGpuBackend Backend;
        public readonly IDisposable GpuOpScope;
        public readonly List<PendingCommit> Pending = new(64);

        public ActiveBatch(DirectGpuTensorEngine engine, IDirectGpuBackend backend)
        {
            Engine = engine;
            Backend = backend;
            GpuOpScope = engine.EnterGpuOp();
        }
    }

    internal readonly struct Scope : IDisposable
    {
        private readonly ActiveBatch? _owned;

        internal Scope(ActiveBatch? owned) => _owned = owned;

        public void Commit()
        {
            if (_owned is { } batch)
            {
                FlushCommits(batch);
            }
        }

        public void Dispose()
        {
            if (_owned is not { } batch) return;
            try
            {
                // Flush any remaining uncommitted device writes so tensors modified before an early return remain coherent.
                FlushCommits(batch);
            }
            finally
            {
                if (ReferenceEquals(_current, batch))
                {
                    _current = null;
                }
                batch.GpuOpScope.Dispose();
            }
        }
    }

    [ThreadStatic]
    private static ActiveBatch? _current;

    private static readonly Func<GpuSyncPoint, int, GpuSyncPoint[]>? SharedLeaseFactory = ResolveSharedLeaseFactory();

    internal static Scope Begin(DirectGpuTensorEngine? gpuEngine)
    {
        if (gpuEngine is null) return default;
        var backend = gpuEngine.GetBackend();
        if (backend is null) return default;
        if (_current is { } existing && ReferenceEquals(existing.Engine, gpuEngine) && ReferenceEquals(existing.Backend, backend))
        {
            // Nested optimizer invocation on the same GPU engine shares the outer scope.
            return default;
        }

        var batch = new ActiveBatch(gpuEngine, backend);
        _current = batch;
        return new Scope(batch);
    }

    internal static void RecordTensorUpdated(IDirectGpuBackend backend, Tensor<float> tensor)
    {
        if (_current is { } batch && ReferenceEquals(batch.Backend, backend))
        {
            batch.Pending.Add(new PendingCommit(tensor, parameterBuffer: null, isParameter: false));
            return;
        }

        ApplyTensorUpdated(tensor, CreateWriteSyncPoint(backend));
    }

    internal static void RecordParameterUpdated(
        DirectGpuTensorEngine gpuEngine,
        IDirectGpuBackend gpuBackend,
        Tensor<float> parameter,
        IGpuBuffer buffer)
    {
        if (_current is { } batch
            && ReferenceEquals(batch.Engine, gpuEngine)
            && ReferenceEquals(batch.Backend, gpuBackend))
        {
            batch.Pending.Add(new PendingCommit(parameter, buffer, isParameter: true));
            return;
        }

        ApplyParameterUpdated(gpuEngine, gpuBackend, parameter, buffer, CreateWriteSyncPoint(gpuBackend));
    }

    private static void FlushCommits(ActiveBatch batch)
    {
        int count = batch.Pending.Count;
        if (count == 0) return;

        try
        {
            if (count == 1)
            {
                CommitSingle(batch.Engine, batch.Backend, batch.Pending[0], CreateWriteSyncPoint(batch.Backend));
                return;
            }

            if (SharedLeaseFactory is { } createGroup)
            {
                var leases = createGroup(CreateWriteSyncPoint(batch.Backend), count);
                if (leases is not null && leases.Length == count)
                {
                    for (int i = 0; i < count; i++)
                    {
                        CommitSingle(batch.Engine, batch.Backend, batch.Pending[i], leases[i]);
                    }
                    return;
                }
            }

            for (int i = 0; i < count; i++)
            {
                CommitSingle(batch.Engine, batch.Backend, batch.Pending[i], CreateWriteSyncPoint(batch.Backend));
            }
        }
        finally
        {
            batch.Pending.Clear();
        }
    }

    private static void CommitSingle(
        DirectGpuTensorEngine engine,
        IDirectGpuBackend backend,
        PendingCommit item,
        GpuSyncPoint syncPoint)
    {
        if (item.IsParameter && item.ParameterBuffer is { } buffer)
        {
            ApplyParameterUpdated(engine, backend, item.Tensor, buffer, syncPoint);
        }
        else
        {
            ApplyTensorUpdated(item.Tensor, syncPoint);
        }
    }

    private static void ApplyTensorUpdated(Tensor<float> tensor, GpuSyncPoint syncPoint)
    {
        tensor.MarkModified(syncPoint);
        tensor._gpuBufferVersion = tensor.Version;
    }

    private static void ApplyParameterUpdated(
        DirectGpuTensorEngine gpuEngine,
        IDirectGpuBackend gpuBackend,
        Tensor<float> pf,
        IGpuBuffer buffer,
        GpuSyncPoint syncPoint)
    {
        if (pf._gpuBuffer is not null ||
            (pf.OffloadDevicePointer == IntPtr.Zero && pf.GetBackingArrayForCacheLookupUnsafe() is not null))
        {
            gpuEngine.CommitParameterUpdatedOnDevice(pf, buffer, gpuBackend, syncPoint);
        }
        else
        {
            ApplyTensorUpdated(pf, syncPoint);
        }
    }

    private static GpuSyncPoint CreateWriteSyncPoint(IDirectGpuBackend backend)
        => backend is IAsyncGpuBackend asyncBackend
            ? asyncBackend.CreateSyncPoint()
            : GpuSyncPoint.CreateComplete();

    private static Func<GpuSyncPoint, int, GpuSyncPoint[]>? ResolveSharedLeaseFactory()
    {
        try
        {
            var method = typeof(GpuSyncPoint).GetMethod(
                "CreateSharedLeaseGroup",
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static,
                binder: null,
                types: new[] { typeof(GpuSyncPoint), typeof(int) },
                modifiers: null);
            if (method is null || method.ReturnType != typeof(GpuSyncPoint[])) return null;
            return (Func<GpuSyncPoint, int, GpuSyncPoint[]>)Delegate.CreateDelegate(
                typeof(Func<GpuSyncPoint, int, GpuSyncPoint[]>), method);
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                $"[AiDotNet] Failed to bind GpuSyncPoint.CreateSharedLeaseGroup: {ex.GetType().Name}: {ex.Message}");
            return null;
        }
    }
}
