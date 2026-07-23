using AiDotNet.Interfaces;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// Integration tests for Issue #1123: verifies that gradient-based optimizers
/// don't redundantly call model.Train() every epoch, and that the content-based
/// cache key in GenerateCacheKey prevents redundant evaluations.
/// </summary>
public class OptimizerTrainSkipTests
{
    [Fact(Timeout = 30000)]
    public async Task ClosedFormModel_WithOptimizer_CompletesQuickly()
    {
        await Task.Yield();
        // A closed-form model (MultipleRegression) through Adam optimizer
        // should NOT take minutes — the cache should prevent redundant Train() calls.
        var rng = new Random(42);
        int n = 500;
        int features = 4;

        var x = new Matrix<double>(n, features);
        var y = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < features; j++)
                x[i, j] = rng.NextDouble() * 10;
            y[i] = 1.0 + 2.0 * x[i, 0] + 3.0 * x[i, 1] + rng.NextDouble() * 0.1;
        }

        var model = new MultipleRegression<double>(new RegressionOptions<double> { UseIntercept = true });
        var optimizerOptions = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 100,
            InitialLearningRate = 0.01
        };
        var optimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(model, optimizerOptions);

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = x, YTrain = y,
            XValidation = x, YValidation = y,
            XTest = x, YTest = y
        };

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = optimizer.Optimize(inputData);
        sw.Stop();

        // Should complete in under 10 seconds, not 5+ minutes
        Assert.True(sw.Elapsed.TotalSeconds < 10,
            $"Optimizer took {sw.Elapsed.TotalSeconds:F1}s for closed-form model — " +
            "redundant Train() calls likely not skipped (Issue #1123).");

        // Result should still be a valid model
        Assert.NotNull(result.BestSolution);
    }

    [Fact(Timeout = 30000)]
    public async Task GradientOptimizer_DoesNotCallTrainDuringEvaluation()
    {
        await Task.Yield();
        // Verify that gradient-based optimizer's EvaluateSolution does NOT
        // call model.Train() — it should only call Predict() for fitness.
        var model = new TrainCountingModel();
        var optimizerOptions = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 10,
            InitialLearningRate = 0.01
        };
        var optimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(model, optimizerOptions);

        var x = new Matrix<double>(50, 3);
        var y = new Vector<double>(50);
        var rng = new Random(42);
        for (int i = 0; i < 50; i++)
        {
            for (int j = 0; j < 3; j++) x[i, j] = rng.NextDouble();
            y[i] = x[i, 0] + x[i, 1];
        }

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = x, YTrain = y,
            XValidation = x, YValidation = y,
            XTest = x, YTest = y
        };

        var result = optimizer.Optimize(inputData);

        // With gradient-based optimizer, Train() should be called at most once
        // (for initial evaluation), NOT once per epoch
        Assert.True(model.TrainCallCount <= 2,
            $"model.Train() was called {model.TrainCallCount} times during 10 epochs — " +
            "gradient-based optimizer should skip Train() during evaluation.");
    }

    [Fact(Timeout = 30000)]
    public async Task CacheKey_SameParameters_ReturnsCachedResult()
    {
        await Task.Yield();
        // When parameters don't change, the cache key should match
        // and TrainAndEvaluateSolution should return cached result.
        var model = new TrainCountingModel();
        var optimizerOptions = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            MaxIterations = 5,
            InitialLearningRate = 0.01
        };
        var optimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(model, optimizerOptions);

        var x = new Matrix<double>(20, 2);
        var y = new Vector<double>(20);
        var rng = new Random(42);
        for (int i = 0; i < 20; i++)
        {
            for (int j = 0; j < 2; j++) x[i, j] = rng.NextDouble();
            y[i] = x[i, 0] * 2;
        }

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = x, YTrain = y,
            XValidation = x, YValidation = y,
            XTest = x, YTest = y
        };

        var result = optimizer.Optimize(inputData);

        // Non-gradient optimizer with no feature selection: same parameters each time
        // should hit cache after first call, so Train() runs very few times
        Assert.True(model.TrainCallCount <= 5,
            $"model.Train() was called {model.TrainCallCount} times for 5 iterations — " +
            "cache should prevent redundant calls when parameters unchanged.");
    }

    /// <summary>
    /// Test model that counts how many times Train() is called.
    /// </summary>
    private class TrainCountingModel : MultipleRegression<double>
    {
        public int TrainCallCount { get; private set; }

        public TrainCountingModel() : base(new RegressionOptions<double> { UseIntercept = true })
        {
        }

        public override void Train(Matrix<double> x, Vector<double> y)
        {
            TrainCallCount++;
            base.Train(x, y);
        }
    }
}
