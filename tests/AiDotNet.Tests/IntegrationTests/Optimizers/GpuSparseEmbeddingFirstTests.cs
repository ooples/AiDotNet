using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Diagnostics;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// On a GPU engine a sparse embedding gradient must still take the optimizer's sparse step first, as Adam and AdamW
/// do: only the looked-up rows are touched. The other optimizers skipped their sparse helper whenever the engine was a
/// GPU engine, so the gradient was densified into a full [vocab, dim] table, uploaded and stepped on the device -- a
/// whole-table pass (and upload) per step for the handful of rows a batch touches.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class GpuSparseEmbeddingFirstTests
{
    private const int Vocab = 4096, Dim = 16;

    public static IEnumerable<object[]> Optimizers() => new[]
    {
        "AMSGrad", "AdaDelta", "AdaMax", "Adagrad", "FTRL", "Lion", "Nadam",
        "SGD", "GradientDescent", "MiniBatchGradientDescent",
        // Momentum is not here: its sparse helper declines any nonzero momentum (dense momentum moves every row).
    }.Select(name => new object[] { name });

    private static IGradientBasedOptimizer<float, Matrix<float>, Vector<float>> Create(string name) => name switch
    {
        "AMSGrad" => new AMSGradOptimizer<float, Matrix<float>, Vector<float>>(null),
        "AdaDelta" => new AdaDeltaOptimizer<float, Matrix<float>, Vector<float>>(null),
        "AdaMax" => new AdaMaxOptimizer<float, Matrix<float>, Vector<float>>(null),
        "Adagrad" => new AdagradOptimizer<float, Matrix<float>, Vector<float>>(null),
        "FTRL" => new FTRLOptimizer<float, Matrix<float>, Vector<float>>(null),
        "Lion" => new LionOptimizer<float, Matrix<float>, Vector<float>>(null),
        "Nadam" => new NadamOptimizer<float, Matrix<float>, Vector<float>>(null),
        "SGD" => new StochasticGradientDescentOptimizer<float, Matrix<float>, Vector<float>>(null),
        "GradientDescent" => new GradientDescentOptimizer<float, Matrix<float>, Vector<float>>(null),
        "MiniBatchGradientDescent" => new MiniBatchGradientDescentOptimizer<float, Matrix<float>, Vector<float>>(null),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    [SkippableTheory]
    [MemberData(nameof(Optimizers))]
    public void SparseEmbeddingGradient_TakesTheSparseStep_OnAGpuEngine(string name)
    {
        // A missing backend is reported through IsGpuAvailable; a constructor exception is a real failure.
        var gpu = new DirectGpuTensorEngine();
        var saved = AiDotNetEngine.Current;
        try
        {
            Skip.If(!gpu.SupportsGpu || !gpu.IsGpuAvailable, "No DirectGpuTensorEngine available.");
            AiDotNetEngine.Current = gpu;
            var optimizer = Create(name);
            var table = new Tensor<float>(new[] { Vocab, Dim });
            for (int i = 0; i < table.Length; i++) table[i] = (float)Math.Sin(0.37 * i) + 0.1f;
            var initial = table.ToArray();
            var ids = new Tensor<int>(new[] { 7, 1500, 3021 }, new[] { 3 });   // distinct rows
            var weight = new Tensor<float>(new[] { ids.Length, Dim });
            for (int i = 0; i < weight.Length; i++) weight[i] = 0.2f + 0.05f * i;

            bool sawSparse = false;
            long uploaded;
            using (var tape = new GradientTape<float>())
            {
                var rows = gpu.TensorEmbeddingLookup<float, int>(table, ids);
                var loss = gpu.ReduceSum(gpu.TensorMultiply(rows, weight), null);
                // Measured from here: the forward's lookup uploads the table itself.
                using var scope = GpuResidencyScope.Begin();
                // Sparse embedding gradients are live only while the backward runs: step inside the streaming callback,
                // as the network's optimizer-in-backward does.
                tape.ComputeGradientsStreaming(loss, new[] { table }, (source, grad) =>
                {
                    sawSparse |= SparseEmbeddingOptimizerHelpers.HasSparseEmbeddingGrad(source);
                    // Unprocessed: only the sparse side channel, no dense entry (a dense entry means clipping or
                    // regularization already produced the gradient the update must use).
                    optimizer.Step(new TapeStepContext<float>(new[] { source },
                        new Dictionary<Tensor<float>, Tensor<float>>(), 0f));
                });
                uploaded = scope.BytesUploaded;
            }

            Assert.True(sawSparse, "the GPU embedding backward produced no sparse gradient, so nothing was exercised");
            var after = table.ToArray();
            var touched = new HashSet<int>(ids.ToArray());
            int moved = 0;
            for (int r = 0; r < Vocab; r++)
                for (int c = 0; c < Dim; c++)
                {
                    int i = r * Dim + c;
                    if (touched.Contains(r)) { if (after[i] != initial[i]) moved++; }
                    else Assert.True(after[i] == initial[i], $"{name}: untouched row {r} changed");
                }
            Assert.True(moved > 0, $"{name}: the looked-up rows did not move");
            // A dense step uploads at least the [vocab, dim] gradient; the sparse one never materializes it.
            long tableBytes = (long)Vocab * Dim * sizeof(float);
            Assert.True(uploaded < tableBytes, $"{name}: {uploaded} bytes uploaded, at least a dense table ({tableBytes})");
        }
        finally
        {
            AiDotNetEngine.Current = saved;
            gpu.Dispose();
        }
    }
}
