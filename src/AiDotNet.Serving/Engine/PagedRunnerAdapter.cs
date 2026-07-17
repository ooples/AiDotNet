using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Engine;

/// <summary>
/// The fast-path model driver: adapts a model's paged <see cref="ICausalLmRunner{T}"/> capability to the
/// engine's <see cref="IServingModelRunner{T}"/>. It splits each scheduled batch into its prefill and decode
/// sub-batches, performs any copy-on-write block copies, invokes the runner's incremental
/// <see cref="ICausalLmRunner{T}.Prefill"/> / <see cref="ICausalLmRunner{T}.DecodeStep"/>, and scatters the
/// per-sequence logits back into scheduling order.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> when a model supports the fast paged path (it keeps a real KV cache), this
/// adapter is what actually calls it: it separates the requests that are still reading their prompt (prefill)
/// from those generating one token at a time (decode), runs each group efficiently against the paged cache,
/// and lines the results back up so the engine can sample.</para>
/// </remarks>
/// <typeparam name="T">The model's numeric type.</typeparam>
public sealed class PagedRunnerAdapter<T> : IServingModelRunner<T>
{
    private readonly ICausalLmRunner<T> _runner;

    /// <summary>Creates the adapter over a paged causal-LM runner.</summary>
    public PagedRunnerAdapter(ICausalLmRunner<T> runner)
    {
        _runner = runner ?? throw new ArgumentNullException(nameof(runner));
        if (_runner.VocabularySize < 1)
            throw new ArgumentException("Runner vocabulary size must be positive.", nameof(runner));
    }

    /// <inheritdoc/>
    public int VocabularySize => _runner.VocabularySize;

    /// <inheritdoc/>
    public IReadOnlyList<Vector<T>> Execute(IReadOnlyList<SequenceExecution<T>> batch)
    {
        if (batch is null) throw new ArgumentNullException(nameof(batch));

        // Perform any engine-requested copy-on-write block copies before writes.
        List<BlockCopy>? copies = null;
        foreach (var exec in batch)
        {
            if (exec.BlockCopies.Count == 0) continue;
            copies ??= new List<BlockCopy>();
            copies.AddRange(exec.BlockCopies);
        }
        if (copies is { Count: > 0 }) _runner.CopyBlocks(copies);

        var results = new Vector<T>[batch.Count];

        var prefillTokens = new List<IReadOnlyList<int>>();
        var prefillLayouts = new List<SequenceKvLayout>();
        var prefillCounts = new List<int>();
        var prefillPos = new List<int>();

        var decodeLast = new List<int>();
        var decodeLayouts = new List<SequenceKvLayout>();
        var decodePos = new List<int>();

        for (int i = 0; i < batch.Count; i++)
        {
            var exec = batch[i];
            var layout = new SequenceKvLayout(exec.SequenceId, exec.BlockTable, exec.NumComputedTokens);
            if (exec.IsPrefill)
            {
                prefillTokens.Add(exec.AllTokenIds);
                prefillLayouts.Add(layout);
                prefillCounts.Add(exec.AllTokenIds.Count - exec.NumComputedTokens);
                prefillPos.Add(i);
            }
            else
            {
                decodeLast.Add(exec.AllTokenIds[exec.AllTokenIds.Count - 1]);
                decodeLayouts.Add(layout);
                decodePos.Add(i);
            }
        }

        if (prefillPos.Count > 0)
        {
            var logits = _runner.Prefill(prefillTokens, prefillLayouts, prefillCounts);
            ScatterRows(logits, prefillPos, results);
        }
        if (decodePos.Count > 0)
        {
            var logits = _runner.DecodeStep(decodeLast, decodeLayouts);
            ScatterRows(logits, decodePos, results);
        }

        return results;
    }

    // Copies each row of a [subBatch, vocab] logits tensor into the result slot for that sequence.
    private void ScatterRows(Tensor<T> logits, List<int> positions, Vector<T>[] results)
    {
        if (logits is null) throw new InvalidOperationException("Runner returned null logits.");
        var shape = logits.Shape;
        if (shape.Length != 2 || shape[0] != positions.Count || shape[1] != _runner.VocabularySize)
            throw new InvalidOperationException(
                $"Runner returned logits of rank {shape.Length} for {positions.Count} sequences; " +
                $"expected [{positions.Count}, {_runner.VocabularySize}].");

        int vocab = _runner.VocabularySize;
        for (int k = 0; k < positions.Count; k++)
        {
            var row = new T[vocab];
            for (int v = 0; v < vocab; v++) row[v] = logits[k, v];
            results[positions[k]] = new Vector<T>(row);
        }
    }
}
