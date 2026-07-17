using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Engine;

/// <summary>
/// The correctness-first model driver: adapts any <see cref="ICausalLmModel{T}"/> to the engine's
/// <see cref="IServingModelRunner{T}"/> by recomputing next-token logits from the full token sequence each
/// step. It needs no KV plumbing, so it works for every generative model out of the box — the universal
/// fallback beneath the optional paged fast path.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> this is the "simple but always-correct" way to run a model in the engine: every
/// step it hands the model all the tokens so far and asks for the next-token scores. It ignores the paged KV
/// cache (that is the fast path's job) and instead recomputes, which is why it pairs naturally with recompute
/// preemption — a recomputing sequence and this runner do exactly the same work.</para>
/// </remarks>
/// <typeparam name="T">The model's numeric type.</typeparam>
public sealed class RecomputeModelRunner<T> : IServingModelRunner<T>
{
    private readonly ICausalLmModel<T> _model;

    /// <summary>Creates a recompute runner over a causal-LM model.</summary>
    public RecomputeModelRunner(ICausalLmModel<T> model)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        if (_model.VocabularySize < 1)
            throw new ArgumentException("Model vocabulary size must be positive.", nameof(model));
    }

    /// <inheritdoc/>
    public int VocabularySize => _model.VocabularySize;

    /// <inheritdoc/>
    public IReadOnlyList<Vector<T>> Execute(IReadOnlyList<SequenceExecution<T>> batch)
    {
        if (batch is null) throw new ArgumentNullException(nameof(batch));
        var numOps = MathHelper.GetNumericOperations<T>();
        var results = new List<Vector<T>>(batch.Count);

        foreach (var exec in batch)
        {
            var tokens = exec.AllTokenIds;
            int n = tokens.Count;
            var input = new Tensor<T>(new[] { 1, n });
            for (int i = 0; i < n; i++)
                input[0, i] = numOps.FromDouble(tokens[i]);

            var logits = _model.ForwardLogits(input);
            results.Add(ExtractLastPositionLogits(logits, _model.VocabularySize));
        }
        return results;
    }

    // Reads the vocabulary logits at the final sequence position, tolerating the common output ranks a model
    // might return: [vocab], [seq, vocab], or [1, seq, vocab] (optionally [1, vocab]).
    private static Vector<T> ExtractLastPositionLogits(Tensor<T> logits, int vocab)
    {
        if (logits is null) throw new InvalidOperationException("Model returned null logits.");
        var shape = logits.Shape;

        switch (shape.Length)
        {
            case 1:
            {
                if (shape[0] != vocab) throw ShapeError(shape, vocab);
                var row = new T[vocab];
                for (int v = 0; v < vocab; v++) row[v] = logits[v];
                return new Vector<T>(row);
            }
            case 2:
            {
                // [seq, vocab] (or [1, vocab]); take the last row.
                if (shape[1] != vocab) throw ShapeError(shape, vocab);
                int last = shape[0] - 1;
                var row = new T[vocab];
                for (int v = 0; v < vocab; v++) row[v] = logits[last, v];
                return new Vector<T>(row);
            }
            case 3:
            {
                // [batch(1), seq, vocab]; take the last position of the first (only) batch item.
                if (shape[2] != vocab) throw ShapeError(shape, vocab);
                int last = shape[1] - 1;
                var row = new T[vocab];
                for (int v = 0; v < vocab; v++) row[v] = logits[0, last, v];
                return new Vector<T>(row);
            }
            default:
                throw ShapeError(shape, vocab);
        }
    }

    private static InvalidOperationException ShapeError(TensorShape shape, int vocab)
    {
        var dims = new int[shape.Length];
        for (int i = 0; i < shape.Length; i++) dims[i] = shape[i];
        return new InvalidOperationException(
            $"Model logits shape [{string.Join(",", dims)}] is not a recognized " +
            $"[vocab] / [seq, vocab] / [1, seq, vocab] layout with vocabulary size {vocab}.");
    }
}
