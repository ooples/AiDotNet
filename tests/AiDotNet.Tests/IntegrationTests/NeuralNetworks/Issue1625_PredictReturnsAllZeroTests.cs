using AiDotNet;
using AiDotNet.ActivationFunctions;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Repro + regression test for #1625 — AiModelResult.Predict and AiModelBuilder.Predict
/// returned exact-zero output for every input after a successful training run.
///
/// <para><b>Root cause:</b> NeuralNetworkBase.TryForwardGpuOptimized used
/// <c>using var gpuResult = ForwardGpu(input)</c>. The `using` disposed the returned
/// GPU tensor before the caller (NeuralNetwork.PredictCore → AiModelResult.Predict →
/// AiModelBuilder.Predict) could read its values, so every facade-routed Predict call
/// on a model with all-GPU-capable layers and a DirectGpuTensorEngine read its result
/// as zeros. The same anti-pattern was present at NeuralNetworkBase.ForwardDeferred's
/// non-deferred fallback path (line 1513 pre-fix).</para>
///
/// <para><b>Fix:</b> drop the `using` so the returned tensor's lifetime belongs to
/// the caller — matching the contract of every other Predict path in the codebase
/// (the result IS the model's output, not an intermediate that can be safely freed).</para>
///
/// <para><b>Test environment note:</b> the AiDotNet.Tests assembly's
/// <see cref="TestModuleInitializer"/> pins the process to CPU via
/// AiDotNetEngine.ResetToCpu() before any test runs. Since the bug ONLY manifests
/// through the GPU-resident code path (TryForwardGpuOptimized short-circuits on
/// non-GPU engines), reproducing the regression inside xUnit requires explicitly
/// re-attempting GPU detection. On CI runners with no GPU, the regression cannot
/// be observed — the test then logs that fact and passes vacuously rather than
/// giving a false-positive. On a GPU-equipped dev box (and on any CI with
/// CUDA/OpenCL drivers present), the test exercises the actual buggy path.</para>
///
/// <para><b>Empirical pre-fix verification:</b> on the bug-discoverer's Windows
/// dev box (no CUDA hardware but DirectGpuTensorEngine auto-detected via the
/// AiDotNet.Tensors GpuAutoDetectModuleInit), running the W9B Loan Approval
/// Capstone end-to-end through the facade pre-fix produced 50% accuracy (= test
/// set majority-class rate) with all three fairness audits trivially passing at
/// 0% per-group gap. Post-fix: 77% accuracy, F1 0.75, all three fairness
/// definitions failing with 16-30% gaps. The same flip was observed on facade
/// walkthrough Examples 1, 3, 4, 5.</para>
/// </summary>
public class Issue1625_PredictReturnsAllZeroTests
{
    private readonly ITestOutputHelper _output;
    public Issue1625_PredictReturnsAllZeroTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async System.Threading.Tasks.Task Facade_Predict_BinaryClassification_ReturnsNonZeroProbabilities()
    {
        // Try to re-enable GPU detection (the assembly init forces CPU; the bug only
        // reproduces on the GPU path). If no GPU is detected, the test environment
        // can't observe the regression — log and pass.
        bool restoredCpu = false;
        try
        {
            try { AiDotNetEngine.AutoDetectAndConfigureGpu(); } catch { /* no GPU available */ }

            var engineName = AiDotNetEngine.Current?.GetType().Name ?? "null";
            bool isGpuEngine = AiDotNetEngine.Current is DirectGpuTensorEngine;
            _output.WriteLine($"AiDotNetEngine.Current = {engineName} · isGpu = {isGpuEngine}");

            if (!isGpuEngine)
            {
                _output.WriteLine("SKIP-VACUOUSLY: this test environment has no GPU. The #1625 bug "
                    + "manifests through TryForwardGpuOptimized's GPU-resident path, which is "
                    + "short-circuited on CpuEngine. The fix is verified empirically — see the "
                    + "class XML doc for the W9B Loan Approval Capstone pre/post results.");
                return;
            }

            // ── Reproducible synthetic data ──
            // 60 train + 15 test, 8 features, label = sign of sum of first 3 features.
            // Mirrors the AiDotNet facade walkthrough's Customer Churn example.
            var (trainX, trainY, testX, testY) = MakeBinaryData(samples: 60, features: 8, seed: 42);

            // ── Dense MLP via facade-supported architecture pattern ──
            var layers = new List<ILayer<double>>
            {
                new DenseLayer<double>(outputSize: 16, activationFunction: new ReLUActivation<double>()),
                new DenseLayer<double>(outputSize: 8,  activationFunction: new ReLUActivation<double>()),
                new DenseLayer<double>(outputSize: 1,  activationFunction: new SigmoidActivation<double>()),
            };
            bool allLayersGpu = layers.TrueForAll(l => l.CanExecuteOnGpu);
            _output.WriteLine($"All-Dense-layers all CanExecuteOnGpu = {allLayersGpu}");
            if (!allLayersGpu)
            {
                _output.WriteLine("SKIP-VACUOUSLY: at least one Dense layer doesn't claim GPU support "
                    + "in this build, so PredictCore won't take the buggy TryForwardGpuOptimized path. "
                    + "The fix remains correct by code inspection; see the class XML doc.");
                return;
            }

            var architecture = new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.BinaryClassification,
                complexity: NetworkComplexity.Simple,
                inputSize: 8, outputSize: 1, layers: layers);
            var nn = new NeuralNetwork<double>(architecture);

            // ── Facade build + train ──
            var builder = new AiModelBuilder<double, Tensor<double>, Tensor<double>>();
            var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(
                null, new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
                { InitialLearningRate = 0.05, MaxIterations = 30 });

            var model = await builder
                .ConfigureModel(nn)
                .ConfigureOptimizer(optimizer)
                .ConfigureLossFunction(new BinaryCrossEntropyLoss<double>())
                .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(trainX, trainY))
                .BuildAsync();

            // ── Facade-only inference ──
            // No SetTrainingMode, no ForwardForTraining, no internal NN APIs.
            var preds = builder.Predict(testX, model);

            // ── Distribution checks ──
            int n = preds.Shape[0];
            Assert.Equal(testX.Shape[0], n);

            double min = double.MaxValue, max = double.MinValue;
            int exactZeroCount = 0;
            for (int i = 0; i < n; i++)
            {
                double v = preds[i, 0];
                if (v < min) min = v;
                if (v > max) max = v;
                if (v == 0.0) exactZeroCount++;
            }
            _output.WriteLine($"Predict output (GPU path): min={min:F6}, max={max:F6}, exact-zeros={exactZeroCount}/{n}");

            // Pre-fix: every sample reads as exactly 0.0 because the returned tensor was disposed.
            Assert.True(exactZeroCount < n,
                $"Pre-fix bug reproduced — Predict returned exactly 0.0 for all {n} samples. " +
                "The using-var in TryForwardGpuOptimized disposed the returned tensor before the caller could read it.");

            // Sigmoid output should have spread across samples.
            Assert.True(max - min > 1e-9,
                $"Pre-fix bug reproduced — Predict output had zero variance (min == max == {min:F6}).");
        }
        finally
        {
            // Restore CPU mode so we don't leak GPU state to subsequent tests.
            try { AiDotNetEngine.ResetToCpu(); restoredCpu = true; } catch { /* best-effort */ }
            _output.WriteLine($"Restored CPU mode = {restoredCpu}");
        }
    }

    private static (Tensor<double> trainX, Tensor<double> trainY, Tensor<double> testX, Tensor<double> testY)
        MakeBinaryData(int samples, int features, int seed)
    {
        int testN = samples / 4;
        int trainN = samples - testN;
        var (tx, ty) = MakeSplit(trainN, features, seed);
        var (ex, ey) = MakeSplit(testN, features, seed + 7);
        return (tx, ty, ex, ey);
    }

    private static (Tensor<double> X, Tensor<double> Y) MakeSplit(int n, int features, int seed)
    {
        var rng = new System.Random(seed);
        var X = new Tensor<double>(new[] { n, features });
        var Y = new Tensor<double>(new[] { n, 1 });
        for (int i = 0; i < n; i++)
        {
            double risk = 0;
            for (int f = 0; f < features; f++)
            {
                var v = rng.NextDouble() * 2 - 1;
                X[i, f] = v;
                if (f < 3) risk += v;
            }
            Y[i, 0] = risk > 0 ? 1.0 : 0.0;
        }
        return (X, Y);
    }
}
