using System;
using AiDotNet.Interfaces;

namespace AiDotNet.Serving.Engine;

/// <summary>
/// Chooses the right <see cref="IServingModelRunner{T}"/> for a model, honoring the engine's layered design:
/// the paged fast path when a model advertises it, otherwise the universal recompute path over any
/// <see cref="ICausalLmModel{T}"/>. This is the seam the facade uses so a beginner never picks a runner.
/// </summary>
public static class ServingRunnerFactory
{
    /// <summary>The runner plus the EOS token id it implies (used to configure engine stop conditions).</summary>
    public readonly struct Selection<T>
    {
        /// <summary>Creates a selection.</summary>
        public Selection(IServingModelRunner<T> runner, int? eosTokenId)
        {
            Runner = runner;
            EosTokenId = eosTokenId;
        }

        /// <summary>The chosen model runner.</summary>
        public IServingModelRunner<T> Runner { get; }

        /// <summary>The model's EOS token id, if any.</summary>
        public int? EosTokenId { get; }
    }

    /// <summary>
    /// Selects a runner for <paramref name="model"/>. A model advertising the paged fast path
    /// (<see cref="ICausalLmRunner{T}"/>) is driven incrementally; any <see cref="ICausalLmModel{T}"/> is driven
    /// by the recompute fallback. Throws a clear error for models that are not text generators.
    /// </summary>
    public static Selection<T> Create<T>(object model)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));

        if (model is ICausalLmRunner<T> paged)
            return new Selection<T>(new PagedRunnerAdapter<T>(paged), EosFromRunner(paged));

        if (model is ICausalLmModel<T> lm)
            return new Selection<T>(new RecomputeModelRunner<T>(lm), lm.EosTokenId);

        throw new NotSupportedException(
            $"Model type '{model.GetType().Name}' cannot generate text: it implements neither the paged " +
            $"ICausalLmRunner<{typeof(T).Name}> fast path nor the ICausalLmModel<{typeof(T).Name}> capability. " +
            "Implement ICausalLmModel (expose VocabularySize, EosTokenId, and ForwardLogits) to enable " +
            "Generate()/Serve().");
    }

    // The paged fast path carries EOS on the model side, not the runner; a model can implement both. When it
    // does, prefer the model's declared EOS.
    private static int? EosFromRunner<T>(ICausalLmRunner<T> runner)
        => runner is ICausalLmModel<T> lm ? lm.EosTokenId : null;
}
