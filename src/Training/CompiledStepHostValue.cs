using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Training;

/// <summary>
/// A tensor a training forward computes on the host from other tensors of the same step (a data-dependent choice:
/// the strongest autocorrelation lags, the most active queries), declared so that a compiled training plan recomputes
/// it on every replay. No gradient flows through it.
/// </summary>
/// <remarks>
/// <para>
/// A compiled plan traces the forward once and replays the graph. A choice made on the host from the step's values
/// (read the scores, sort them, keep the top few) is not an engine op, so the trace keeps the first step's answer as a
/// constant and every later step reuses it: the model attends to step one's queries, or aggregates step one's lags,
/// for the whole run. Nothing fails; the model trains on the wrong graph.
/// </para>
/// <para>
/// Computing the choice through this type records it as a node of the plan that reads its inputs' current values and
/// computes again on every replay, as it does eagerly. Use the result through ops that read their indices at run time
/// (masks, matrix products, <c>TensorEmbeddingLookupFromFloatIndices</c>), never through an index tensor built on the
/// host, which the trace would freeze in turn. Per-step random draws belong in <see cref="CompiledStepRandom{T}"/>.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type of the tensors.</typeparam>
public static class CompiledStepHostValue<T>
{
    /// <summary>
    /// The tensor of <paramref name="shape"/> that <paramref name="compute"/> writes from the current values of
    /// <paramref name="inputs"/>; under a compiled trace, a node that recomputes it on every replay.
    /// </summary>
    /// <param name="name">The node's name in the plan (diagnostics).</param>
    /// <param name="inputs">The tensors the value is computed from.</param>
    /// <param name="shape">The shape of the value.</param>
    /// <param name="compute">Writes the value into its second argument, in place, from the first argument's
    /// tensors (the inputs, in order). It must read nothing else that changes between steps.</param>
    public static Tensor<T> Compute(string name, Tensor<T>[] inputs, int[] shape, Action<Tensor<T>[], Tensor<T>> compute)
    {
        Guard.NotNull(name);
        Guard.NotNull(inputs);
        Guard.NotNull(shape);
        Guard.NotNull(compute);
        if (inputs.Length == 0)
            throw new ArgumentException("A host value is computed from at least one tensor of the step.", nameof(inputs));

        if (!GraphMode.IsActive)
        {
            var value = new Tensor<T>(shape);
            compute(inputs, value);
            return value;
        }

        var scope = GraphMode.Current
            ?? throw new InvalidOperationException("GraphMode reported active without a scope.");
        var captured = (Tensor<T>[])inputs.Clone();
        return scope.RecordVariadic(
            LazyNodeType.Custom,
            name,
            captured,
            (int[])shape.Clone(),
            (replayEngine, result) =>
            {
                using var noGrad = new NoGradScope<T>();
                var value = new Tensor<T>(result._shape);
                compute(captured, value);
                AiDotNet.Tensors.Engines.DirectGpuTensorEngine.CopyResultInto(replayEngine, value, result);
            },
            // A choice, not a differentiable function of its inputs: nothing to accumulate.
            static (_, _, _, _, _, _) => { });
    }
}
