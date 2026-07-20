using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.TimeSeries;

/// <summary>
/// Regression cover for the Autoformer training crash surfaced by a research sweep:
///
///   ArgumentException: Tensor shapes must match. Got [2048, 512] and [512, 2048].
///     at CpuEngine.TensorAdd
///     at AutoformerModel`1.TrainCore
///
/// Those two shapes are exactly the model's own FFN weights at the default EmbeddingDim of 512:
/// _ff1Weight is [ffDim, embeddingDim] = [2048, 512] and _ff2Weight is [embeddingDim, ffDim] =
/// [512, 2048] (ffDim = 4 * embeddingDim). TrainCore accumulates per-sample gradients with
/// accum[param] = TensorAdd(acc, g), so the throw means one FFN weight is receiving the OTHER's
/// gradient shape — the two weights' gradients are crossed.
///
/// The forward is not obviously at fault: both use TensorMatMul(x, TensorTranspose(w)), which is
/// dimensionally correct. That points at the transpose backward failing to transpose the incoming
/// gradient back before attributing it to the parameter.
///
/// This test uses small dims (embeddingDim 8 => ffDim 32) so the same crossing appears as
/// [32, 8] vs [8, 32] and runs in seconds instead of minutes. It asserts only that training
/// COMPLETES — a shape-crossed gradient throws, so completion is the signal.
/// </summary>
public class AutoformerTrainShapeReproTests
{
    private static (Matrix<double> X, Vector<double> Y) SyntheticSeries(int n, int features, int seed = 11)
    {
        var rng = new Random(seed);
        var x = new Matrix<double>(n, features);
        var y = new Vector<double>(n);
        double level = 0.0;
        for (int i = 0; i < n; i++)
        {
            // Mean-reverting series so the model has real (not degenerate) signal to fit.
            level = 0.6 * level + (rng.NextDouble() - 0.5) * 0.02;
            y[i] = level;
            for (int f = 0; f < features; f++)
            {
                x[i, f] = i > f ? y[i - f - 1] : 0.0;
            }
        }

        return (x, y);
    }

    // embeddingDim 8 (ffDim 32) passes, so this is NOT a universal FFN crossing — it is dimension
    // dependent. BackwardFunctions carries size-thresholded specialised paths, so the real default of
    // 512 (ffDim 2048) is tested explicitly rather than assumed to behave like the small case.
    [Theory]
    [InlineData(8)]
    [InlineData(64)]
    [InlineData(512)]   // the production default — the configuration that crashed the sweep
    public void Autoformer_trains_without_crossing_its_ffn_gradient_shapes(int embeddingDim)
    {
        var (x, y) = SyntheticSeries(n: 160, features: 3);

        var options = new AutoformerOptions<double>
        {
            LookbackWindow = 12,
            ForecastHorizon = 1,
            // ffDim is 4 * EmbeddingDim. A crossed gradient shows up as "Got [4d, d] and [d, 4d]" —
            // e.g. [32, 8] at d=8, or [2048, 512] at the default d=512 seen in the sweep.
            EmbeddingDim = embeddingDim,
            NumEncoderLayers = 1,
            NumDecoderLayers = 1,
            Epochs = 2,
            UseEarlyStopping = false,
        };

        var model = new AutoformerModel<double>(options);

        // A crossed FFN gradient throws ArgumentException out of TrainCore's accumulation step.
        var ex = Record.Exception(() => model.Train(x, y));

        Assert.True(
            ex is null,
            $"Autoformer training threw {ex?.GetType().Name}: {ex?.Message}. " +
            "A 'Tensor shapes must match' here means the FFN weights' gradients are crossed " +
            "(ff1 is [ffDim, embDim], ff2 is [embDim, ffDim]).");
    }

    /// <summary>
    /// The sweep crashed on AutoformerModel&lt;FLOAT&gt;, and float is precisely the type that unlocks the
    /// resident/fused specialised paths in the engine (roughly 143 typeof(T)==float gates) which a double
    /// model never touches. The double cases above all pass at the same dimensions, so this reproduces the
    /// sweep's exact configuration: float, LookbackWindow 24, default EmbeddingDim/layer counts.
    /// </summary>
    [Fact]
    public void Autoformer_float_trains_at_the_sweep_configuration()
    {
        var (xd, yd) = SyntheticSeries(n: 200, features: 3);
        var x = new Matrix<float>(xd.Rows, xd.Columns);
        for (int i = 0; i < xd.Rows; i++)
            for (int j = 0; j < xd.Columns; j++)
                x[i, j] = (float)xd[i, j];
        var y = new Vector<float>(yd.Length);
        for (int i = 0; i < yd.Length; i++) y[i] = (float)yd[i];

        // Mirrors Ooples' AutoformerFloatForecaster exactly: only these options are set, so
        // EmbeddingDim (512) and the encoder/decoder layer counts stay at their defaults.
        var options = new AutoformerOptions<float>
        {
            LookbackWindow = 24,
            ForecastHorizon = 1,
            Epochs = 2,
            UseEarlyStopping = true,
            EarlyStoppingPatience = 5,
        };

        var model = new AutoformerModel<float>(options);
        var ex = Record.Exception(() => model.Train(x, y));

        Assert.True(
            ex is null,
            $"Autoformer<float> training threw {ex?.GetType().Name}: {ex?.Message}. " +
            "Expected shapes of the form [2048, 512] vs [512, 2048] if the FFN gradients are crossed.");
    }

    /// <summary>
    /// The sweep trained THREE cells CONCURRENTLY (OOPLES_RESEARCH_PARALLELISM=3) against one
    /// process-global engine. Two different Autoformer instances each own an _ff1Weight [ffDim, embDim]
    /// and an _ff2Weight [embDim, ffDim], so any process-global state keyed such that instances collide
    /// would add one instance's ff1 gradient to another's ff2 — producing exactly
    /// "Got [2048, 512] and [512, 2048]", and only intermittently (timing dependent), which matches the
    /// sweep crashing one cell while others merely ran slow.
    ///
    /// Every single-threaded configuration passes (dims 8/64/512, double and float, CPU engine and GPU
    /// engine), so concurrency is the remaining untested differentiator. Small dims are used deliberately:
    /// a cross-instance collision does not depend on size, so this stays fast.
    /// </summary>
    [Fact]
    public void Autoformer_concurrent_instances_do_not_cross_gradients()
    {
        const int instances = 3;
        var errors = new System.Collections.Concurrent.ConcurrentBag<Exception>();

        Parallel.For(0, instances, i =>
        {
            try
            {
                var (x, y) = SyntheticSeries(n: 160, features: 3, seed: 11 + i);
                var options = new AutoformerOptions<double>
                {
                    LookbackWindow = 12,
                    ForecastHorizon = 1,
                    EmbeddingDim = 8,
                    NumEncoderLayers = 1,
                    NumDecoderLayers = 1,
                    Epochs = 2,
                    UseEarlyStopping = false,
                };
                new AutoformerModel<double>(options).Train(x, y);
            }
            catch (Exception ex)
            {
                errors.Add(ex);
            }
        });

        Assert.True(
            errors.IsEmpty,
            "Concurrent Autoformer training threw: " +
            string.Join(" | ", errors.Select(e => e.GetType().Name + ": " + e.Message)));
    }

    /// <summary>
    /// The sweep's ACTUAL condition: concurrency AND the GPU engine AND float, together. Every simpler
    /// combination now passes — dims 8/64/512, double and float, CPU engine, GPU engine single-threaded,
    /// and concurrency on the CPU engine. This is the last combination short of real FCHL data.
    /// Dims are reduced to 64 because a cross-instance state collision does not depend on size, and 512
    /// costs ~8 minutes per instance.
    /// </summary>
    [SkippableFact]
    public void Autoformer_concurrent_float_on_gpu_does_not_cross_gradients()
    {
        var restore = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        try
        {
            var gpuEngine = new AiDotNet.Tensors.Engines.DirectGpuTensorEngine();
            Skip.IfNot(gpuEngine.SupportsGpu, "no CUDA GPU available on this host");
            AiDotNet.Tensors.Engines.AiDotNetEngine.Current = gpuEngine;

            var errors = new System.Collections.Concurrent.ConcurrentBag<Exception>();
            Parallel.For(0, 3, i =>
            {
                try
                {
                    var (xd, yd) = SyntheticSeries(n: 160, features: 3, seed: 11 + i);
                    var x = new Matrix<float>(xd.Rows, xd.Columns);
                    for (int r = 0; r < xd.Rows; r++)
                        for (int c = 0; c < xd.Columns; c++)
                            x[r, c] = (float)xd[r, c];
                    var y = new Vector<float>(yd.Length);
                    for (int r = 0; r < yd.Length; r++) y[r] = (float)yd[r];

                    var options = new AutoformerOptions<float>
                    {
                        LookbackWindow = 24,
                        ForecastHorizon = 1,
                        EmbeddingDim = 64,
                        Epochs = 2,
                        UseEarlyStopping = false,
                    };
                    new AutoformerModel<float>(options).Train(x, y);
                }
                catch (Exception ex)
                {
                    errors.Add(ex);
                }
            });

            Assert.True(
                errors.IsEmpty,
                "Concurrent float Autoformer on GPU threw: " +
                string.Join(" | ", errors.Select(e => e.GetType().Name + ": " + e.Message)));
        }
        finally
        {
            AiDotNet.Tensors.Engines.AiDotNetEngine.Current = restore;
        }
    }

    /// <summary>
    /// The sweep ran with the GPU engine ADOPTED, and that is the last untested differentiator: the CPU
    /// cases above all pass. DirectGpuTensorEngine derives from CpuEngine, so the CpuEngine.TensorAdd frame
    /// in the crash stack is exactly what a GPU-engine fallback looks like, and float unlocks the resident /
    /// fused specialisations that a CPU-engine run never enters.
    /// Skips (rather than fails) when no GPU is present so CI stays green on GPU-less runners.
    /// </summary>
    [SkippableFact]
    public void Autoformer_float_trains_with_the_gpu_engine_adopted()
    {
        var restore = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var restoreLogger = AiDotNet.Tensors.Engines.AiDotNetEngine.Logger;
        bool gpu = false;
        var log = new System.Text.StringBuilder();
        try
        {
            // Capture the engine's OWN reason rather than inferring from the bool: it distinguishes
            // "no device" from "detected but failed its correctness probe" from "init threw", and those
            // three point at completely different fixes.
            AiDotNet.Tensors.Engines.AiDotNetEngine.Logger = m => log.AppendLine(m);

            // Construct the GPU engine DIRECTLY rather than going through AutoDetectAndConfigureGpu.
            // AutoDetect additionally runs a correctness probe and allocator registration, either of which
            // can veto adoption — so it cannot distinguish "no device" from "device rejected". Building the
            // engine and reading SupportsGpu isolates device availability, which is what this test needs.
            var gpuEngine = new AiDotNet.Tensors.Engines.DirectGpuTensorEngine();
            gpu = gpuEngine.SupportsGpu;
            Skip.IfNot(gpu, $"DirectGpuTensorEngine reports no GPU. Engine log: {log.ToString().Trim()}");
            AiDotNet.Tensors.Engines.AiDotNetEngine.Current = gpuEngine;

            var (xd, yd) = SyntheticSeries(n: 200, features: 3);
            var x = new Matrix<float>(xd.Rows, xd.Columns);
            for (int i = 0; i < xd.Rows; i++)
                for (int j = 0; j < xd.Columns; j++)
                    x[i, j] = (float)xd[i, j];
            var y = new Vector<float>(yd.Length);
            for (int i = 0; i < yd.Length; i++) y[i] = (float)yd[i];

            var options = new AutoformerOptions<float>
            {
                LookbackWindow = 24,
                ForecastHorizon = 1,
                Epochs = 2,
                UseEarlyStopping = true,
                EarlyStoppingPatience = 5,
            };

            var model = new AutoformerModel<float>(options);
            var ex = Record.Exception(() => model.Train(x, y));

            Assert.True(
                ex is null,
                $"Autoformer<float> on the GPU engine threw {ex?.GetType().Name}: {ex?.Message}. " +
                "Shapes like [2048, 512] vs [512, 2048] mean the FFN weights' gradients are crossed.");
        }
        finally
        {
            AiDotNet.Tensors.Engines.AiDotNetEngine.Logger = restoreLogger;
            AiDotNet.Tensors.Engines.AiDotNetEngine.Current = restore;
        }
    }
}
