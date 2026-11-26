using System.Collections.Concurrent;
using System.Reflection;
using System.Runtime.ExceptionServices;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Optimizers;

/// <summary>
/// Shared whole-parameter-list ("foreach") GPU step for the gradient-based optimizers' tape step.
/// </summary>
/// <typeparam name="T">The optimizer's numeric type; the GPU list steps run only for <see cref="float"/>.</typeparam>
internal static class GpuListOptimizerStep<T>
{
    /// <summary>
    /// Returns the optimizer state tensors one parameter's GPU list step reads and writes (in the order the list step
    /// takes them), creating them exactly as the optimizer's per-parameter loop would, or null to leave that parameter
    /// to the loop.
    /// </summary>
    internal delegate Tensor<T>[]? StateResolver(Tensor<T> param);

    /// <summary>
    /// Returns <paramref name="param"/>'s state in <paramref name="states"/>, creating it with
    /// <paramref name="create"/> on first use, the way each optimizer's per-parameter loop does.
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
    /// operation per parameter. Returns the parameters it updated, which the caller's per-parameter loop must skip, or
    /// null when nothing was updated (the list step is not in this Tensors build, no parameter was eligible, or the
    /// list step declined), in which case the loop runs unchanged.
    /// </summary>
    /// <remarks>
    /// <para>The list step is all-or-nothing: it resolves every buffer before running any kernel, so a decline leaves
    /// every parameter and state untouched and the caller's loop is exactly the per-parameter path. A parameter with an
    /// unprocessed sparse-embedding gradient is always left to the loop, which owns the sparse scatter path.</para>
    /// <para>Only call this when the caller's per-parameter GPU step would be attempted for every parameter (its GPU
    /// gate holds for the whole step), since an eligible parameter here is one whose per-parameter GPU step would run.</para>
    /// </remarks>
    /// <param name="context">The tape step being applied.</param>
    /// <param name="engine">The optimizer's engine (resolves sparse-embedding gradients to dense).</param>
    /// <param name="listStepName">The <c>GpuOptimizer</c> list method, e.g. <c>TryRmspropStepMultiTensor</c>.</param>
    /// <param name="stateCount">How many state tensors per parameter the list step takes.</param>
    /// <param name="resolveStates">Resolves (creating on first use) one parameter's state tensors.</param>
    /// <param name="scalarArguments">The list step's trailing scalar arguments, boxed with their exact types.</param>
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

        var arguments = new object[2 + stateCount + scalarArguments.Length];
        arguments[0] = parameters;
        arguments[1] = gradients;
        for (int s = 0; s < stateCount; s++) arguments[2 + s] = states[s];
        Array.Copy(scalarArguments, 0, arguments, 2 + stateCount, scalarArguments.Length);

        return GpuListStepBridge.Invoke(method, arguments) ? updated : null;
    }
}

/// <summary>
/// Binds <c>GpuOptimizer</c>'s whole-parameter-list optimizer steps by reflection, so this library runs against a
/// Tensors package with or without them: a step the package lacks resolves to null and the optimizer keeps its
/// per-parameter GPU path.
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

    internal static bool Invoke(MethodInfo method, object[] arguments)
    {
        try
        {
            return method.Invoke(null, arguments) is true;
        }
        catch (TargetInvocationException ex) when (ex.InnerException is { } inner)
        {
            // An argument error inside the list step is a caller bug; surface it as itself, not as a reflection wrapper.
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
