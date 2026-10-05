using System;
using System.Collections.Generic;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// The sparse embedding step of Nadam must match the dense step it short-circuits: the Nesterov momentum term is
/// bias-corrected with bc1(t+1) and the gradient term with bc1(t). The sparse helper used bc1(t) for both.
/// </summary>
public class NadamSparseEmbeddingParityTests
{
    private const int Vocab = 6, Dim = 3;

    private static NadamOptimizer<double, Matrix<double>, Vector<double>> NewNadam() =>
        new(null, new NadamOptimizerOptions<double, Matrix<double>, Vector<double>> { InitialLearningRate = 0.05 });

    private static Tensor<double> Table()
    {
        var table = new Tensor<double>(new[] { Vocab, Dim });
        for (int i = 0; i < table.Length; i++) table[i] = Math.Sin(0.7 * i) + 0.1;
        return table;
    }

    [Fact]
    public void Sparse_embedding_step_matches_the_dense_step()
    {
        var engine = new CpuEngine();
        // Distinct rows: the sparse helper declines duplicates and would fall back to the dense step.
        var ids = new Tensor<int>(new[] { 1, 4, 2 }, new[] { 3 });

        // Sparse run: gradients come from a real embedding backward, so the optimizer takes the sparse helper.
        var sparseTable = Table();
        var sparseNadam = NewNadam();
        // Dense run: the same gradient handed over as an ordinary dense tensor.
        var denseTable = Table();
        var denseNadam = NewNadam();

        bool sawSparse = false;
        for (int step = 0; step < 3; step++)
        {
            var weight = new Tensor<double>(new[] { ids.Length, Dim });
            for (int i = 0; i < weight.Length; i++) weight[i] = 0.2 + 0.05 * i;
            using (var tape = new GradientTape<double>())
            {
                // TensorEmbeddingLookup is the op EmbeddingLayer uses and the one whose backward emits sparse grads.
                var rows = engine.TensorEmbeddingLookup<double, int>(sparseTable, ids);
                var loss = engine.ReduceSum(engine.TensorMultiply(rows, weight), null);
                // Sparse embedding gradients are live only while the backward runs, which is where the network's
                // optimizer-in-backward steps; take the Nadam step inside the streaming callback the same way.
                tape.ComputeGradientsStreaming(loss, new[] { sparseTable }, (source, grad) =>
                {
                    sawSparse |= SparseEmbeddingOptimizerHelpers.HasSparseEmbeddingGrad(source);
                    sparseNadam.Step(new TapeStepContext<double>(new[] { source },
                        new Dictionary<Tensor<double>, Tensor<double>> { [source] = grad }, 0.0));
                });
            }

            // The dense equivalent: d(loss)/d(table[id]) accumulates the weight row for every occurrence.
            var denseGrad = new Tensor<double>(new[] { Vocab, Dim });
            for (int r = 0; r < ids.Length; r++)
                for (int c = 0; c < Dim; c++)
                    denseGrad[ids[r], c] += weight[r * Dim + c];
            denseNadam.Step(new TapeStepContext<double>(new[] { denseTable },
                new Dictionary<Tensor<double>, Tensor<double>> { [denseTable] = denseGrad }, 0.0));
        }

        Assert.True(sawSparse, "no sparse embedding gradient was produced, so the sparse Nadam path was never exercised");
        // Positive control: the embedding table actually moved, so both runs trained.
        Assert.NotEqual(Table()[1 * Dim], sparseTable[1 * Dim]);
        for (int i = 0; i < denseTable.Length; i++)
            Assert.Equal(denseTable[i], sparseTable[i], 12);
    }
}