using System;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

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
    /// Selects a runner for <paramref name="model"/>. Preference order: the paged fast path
    /// (<see cref="ICausalLmRunner{T}"/>), then the recompute fallback over <see cref="ICausalLmModel{T}"/>, then
    /// — when <paramref name="vocabularySize"/> is supplied — a Predict-based adapter over any
    /// <see cref="IFullModel{T, TInput, TOutput}"/> whose forward output is vocabulary-width. Throws a clear
    /// error for models that are not text generators.
    /// </summary>
    /// <param name="model">The model to serve.</param>
    /// <param name="vocabularySize">Vocabulary size, enabling the Predict-adapter fallback for models that do
    /// not implement <see cref="ICausalLmModel{T}"/>. Ignored when the model implements a capability directly.</param>
    /// <param name="eosTokenId">EOS token id for the Predict-adapter fallback.</param>
    public static Selection<T> Create<T>(object model, int? vocabularySize = null, int? eosTokenId = null)
    {
        if (model is null) throw new ArgumentNullException(nameof(model));

        if (model is ICausalLmRunner<T> paged)
            return new Selection<T>(new PagedRunnerAdapter<T>(paged), EosFromRunner(paged));

        if (model is ICausalLmModel<T> lm)
            return new Selection<T>(new RecomputeModelRunner<T>(lm), lm.EosTokenId);

        // Predict-adapter fallback: any full model whose forward produces vocab-width logits, given its vocab.
        if (vocabularySize is { } vocab && model is IFullModel<T, Tensor<T>, Tensor<T>> full)
        {
            var adapter = new PredictCausalLmAdapter<T>(full.Predict, vocab, eosTokenId);
            return new Selection<T>(new RecomputeModelRunner<T>(adapter), eosTokenId);
        }

        throw new NotSupportedException(
            $"Model type '{model.GetType().Name}' cannot generate text: it implements neither the paged " +
            $"ICausalLmRunner<{typeof(T).Name}> fast path nor the ICausalLmModel<{typeof(T).Name}> capability. " +
            $"Implement ICausalLmModel (VocabularySize, EosTokenId, ForwardLogits), or — if it is an " +
            $"IFullModel<{typeof(T).Name}, Tensor, Tensor> that outputs vocab-width logits — supply a vocabulary " +
            "size to use the Predict-based adapter.");
    }

    // The paged fast path carries EOS on the model side, not the runner; a model can implement both. When it
    // does, prefer the model's declared EOS.
    private static int? EosFromRunner<T>(ICausalLmRunner<T> runner)
        => runner is ICausalLmModel<T> lm ? lm.EosTokenId : null;
}
