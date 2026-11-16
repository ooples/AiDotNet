using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Training;

/// <summary>
/// A tensor a training forward produces with an engine op that must run again on every step (a fresh dropout mask
/// drawn by the device's stateless generator), declared so that a compiled training plan re-runs the op on every
/// replay instead of freezing the first step's value. No gradient flows through it.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="CompiledStepRandom{T}"/> redraws on the host and uploads every replay, and
/// <see cref="CompiledStepHostValue{T}"/> reads its inputs back to the host; both put a host round trip into every
/// GPU step. A value this type produces stays where the engine computed it: on a GPU engine the replayed result is
/// aliased into the plan's tensor without touching the host.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type of the tensors.</typeparam>
internal static class CompiledStepDeviceValue<T>
{
    /// <summary>
    /// The tensor <paramref name="produce"/> returns; under a compiled trace, a node that calls it again on every
    /// replay with the replaying engine.
    /// </summary>
    /// <param name="name">The node's name in the plan (diagnostics).</param>
    /// <param name="dependsOn">A tensor of the step the value is ordered after (the plan needs at least one input).</param>
    /// <param name="shape">The shape <paramref name="produce"/> returns.</param>
    /// <param name="produce">Computes the value with the given engine. It must return a tensor of
    /// <paramref name="shape"/> and read nothing from the step but its own state (a seed source).</param>
    /// <param name="readsHostStateOnReplay">True when <paramref name="produce"/> reads host state that changes between
    /// replays (a seed it draws itself), which a captured device graph cannot replay; false when everything that
    /// changes reaches it through device tensors (a seed tensor redrawn by <see cref="CompiledStepRandom{T}"/>).</param>
    public static Tensor<T> Compute(string name, Tensor<T> dependsOn, int[] shape, Func<IEngine, Tensor<T>> produce,
        bool readsHostStateOnReplay = true)
    {
        Guard.NotNull(name);
        Guard.NotNull(dependsOn);
        Guard.NotNull(shape);
        Guard.NotNull(produce);

        if (!GraphMode.IsActive)
            return produce(AiDotNetEngine.Current);

        var scope = GraphMode.Current
            ?? throw new InvalidOperationException("GraphMode reported active without a scope.");
        var recorded = scope.RecordVariadic(
            LazyNodeType.Custom,
            name,
            new[] { dependsOn },
            (int[])shape.Clone(),
            (replayEngine, result) =>
            {
                using var noGrad = new NoGradScope<T>();
                var value = produce(replayEngine);
                AiDotNet.Tensors.Engines.DirectGpuTensorEngine.CopyResultInto(replayEngine, value, result);
            },
            static (_, _, _, _, _, _) => { });
        // A value that reads host state on replay (a seed drawn by the closure) would be frozen at its capture-time
        // value by a captured device graph, so such a step is replayed node by node instead.
        if (readsHostStateOnReplay && recorded.LazySource is LazyNode<T> node) node.ReplaysHostState = true;
        return recorded;
    }
}
