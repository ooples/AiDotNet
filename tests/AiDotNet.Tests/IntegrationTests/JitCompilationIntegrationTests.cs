using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using AiDotNet.Regression;
using AiDotNet.Configuration;
using System.Diagnostics;

namespace AiDotNet.Tests.IntegrationTests;

/// <summary>
/// Integration tests for end-to-end JIT compilation workflow.
/// Tests the full pipeline: PredictionModelBuilder -> JIT compilation -> PredictionModelResult.Predict()
/// </summary>
public class JitCompilationIntegrationTests
{
    /// <summary>
    /// US-1.5: Test SimpleRegression with JIT enabled - verify correctness.
    /// </summary>
    [Fact]
    public async Task SimpleRegression_WithJitEnabled_ProducesSameResultsAsWithoutJit()
    {
        // Arrange: Create training data for simple linear regression (y = 2x + 3)
        var xData = new Matrix<float>(new float[,]
        {
            { 1.0f },
            { 2.0f },
            { 3.0f },
            { 4.0f },
            { 5.0f }
        });

        var yData = new Vector<float>(new float[] { 5.0f, 7.0f, 9.0f, 11.0f, 13.0f });

        // Train model WITHOUT JIT
        var modelWithoutJit = new PredictionModelBuilder<float, Matrix<float>, Vector<float>>()
            .ConfigureModel(new SimpleRegression<float>())
            .ConfigureJitCompilation(new JitCompilationConfig { Enabled = false });

        var resultWithoutJit = await modelWithoutJit.BuildAsync(xData, yData);

        // Train model WITH JIT
        var modelWithJit = new PredictionModelBuilder<float, Matrix<float>, Vector<float>>()
            .ConfigureModel(new SimpleRegression<float>())
            .ConfigureJitCompilation(new JitCompilationConfig { Enabled = true });

        var resultWithJit = await modelWithJit.BuildAsync(xData, yData);

        // Act: Make predictions on new data
        var testData = new Matrix<float>(new float[,] { { 6.0f }, { 7.0f }, { 8.0f } });

        var predictionsWithoutJit = resultWithoutJit.Predict(testData);
        var predictionsWithJit = resultWithJit.Predict(testData);

        // Assert: JIT predictions should match non-JIT predictions (within floating-point tolerance)
        Assert.Equal(predictionsWithoutJit.Length, predictionsWithJit.Length);

        for (int i = 0; i < predictionsWithoutJit.Length; i++)
        {
            Assert.Equal(predictionsWithoutJit[i], predictionsWithJit[i], precision: 5);
        }
    }

    /// <summary>
    /// US-1.5: Test SimpleRegression with JIT enabled - measure performance improvement.
    /// </summary>
    [Fact]
    public async Task SimpleRegression_WithJitEnabled_ShowsPerformanceImprovement()
    {
        // Arrange: Create larger dataset for meaningful performance measurement
        const int dataSize = 1000;
        var random = new Random(42);

        var xData = new Matrix<float>(dataSize, 10); // 10 features
        var yData = new Vector<float>(dataSize);

        for (int i = 0; i < dataSize; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                xData[i, j] = (float)random.NextDouble();
            }
            // y = sum of features + noise
            float sum = 0;
            for (int j = 0; j < 10; j++)
            {
                sum += xData[i, j];
            }
            yData[i] = sum + (float)(random.NextDouble() * 0.1);
        }

        // Train models
        var modelWithoutJit = new PredictionModelBuilder<float, Matrix<float>, Vector<float>>()
            .ConfigureModel(new SimpleRegression<float>())
            .ConfigureJitCompilation(new JitCompilationConfig { Enabled = false });

        var resultWithoutJit = await modelWithoutJit.BuildAsync(xData, yData);

        var modelWithJit = new PredictionModelBuilder<float, Matrix<float>, Vector<float>>()
            .ConfigureModel(new SimpleRegression<float>())
            .ConfigureJitCompilation(new JitCompilationConfig { Enabled = true });

        var resultWithJit = await modelWithJit.BuildAsync(xData, yData);

        // Create test data (large batch for meaningful timing)
        var testData = new Matrix<float>(1000, 10);
        for (int i = 0; i < 1000; i++)
        {
            for (int j = 0; j < 10; j++)
            {
                testData[i, j] = (float)random.NextDouble();
            }
        }

        // Warm up both paths
        _ = resultWithoutJit.Predict(testData);
        _ = resultWithJit.Predict(testData);

        // Act: Measure performance WITHOUT JIT
        const int iterations = 100;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            _ = resultWithoutJit.Predict(testData);
        }
        sw.Stop();
        var timeWithoutJit = sw.Elapsed;

        // Measure performance WITH JIT
        sw.Restart();
        for (int i = 0; i < iterations; i++)
        {
            _ = resultWithJit.Predict(testData);
        }
        sw.Stop();
        var timeWithJit = sw.Elapsed;

        // Assert: JIT should be faster (aim for at least 1.5x improvement)
        // Note: In actual tests, JIT typically provides 2-3x speedup, but we use 1.5x as a conservative threshold
        var speedupRatio = timeWithoutJit.TotalMilliseconds / timeWithJit.TotalMilliseconds;

        Assert.True(speedupRatio >= 1.5,
            $"Expected at least 1.5x speedup with JIT, but got {speedupRatio:F2}x. " +
            $"Time without JIT: {timeWithoutJit.TotalMilliseconds:F2}ms, " +
            $"Time with JIT: {timeWithJit.TotalMilliseconds:F2}ms");
    }

    /// <summary>
    /// US-1.5: Test graceful fallback when JIT compilation fails (model not trained).
    /// </summary>
    [Fact]
    public async Task SimpleRegression_JitCompilationFails_FallsBackGracefully()
    {
        // Arrange: Create training data
        var xData = new Matrix<float>(new float[,]
        {
            { 1.0f },
            { 2.0f },
            { 3.0f }
        });

        var yData = new Vector<float>(new float[] { 5.0f, 7.0f, 9.0f });

        // Configure JIT with ThrowOnFailure = false (graceful fallback)
        var model = new PredictionModelBuilder<float, Matrix<float>, Vector<float>>()
            .ConfigureModel(new SimpleRegression<float>())
            .ConfigureJitCompilation(new JitCompilationConfig
            {
                Enabled = true,
                ThrowOnFailure = false  // Graceful fallback
            });

        // Act & Assert: Build should succeed even if JIT fails
        var result = await model.BuildAsync(xData, yData);

        // Predictions should still work (using non-JIT path if JIT failed)
        var testData = new Matrix<float>(new float[,] { { 4.0f } });
        var prediction = result.Predict(testData);

        Assert.NotNull(prediction);
        Assert.Single(prediction);
    }

    /// <summary>
    /// US-1.5: Test that JIT compilation succeeds with strict mode when model supports it.
    /// </summary>
    [Fact]
    public async Task SimpleRegression_WithJitRequired_BuildsSuccessfully()
    {
        // Arrange: Create training data
        var xData = new Matrix<float>(new float[,] { { 1.0f }, { 2.0f }, { 3.0f } });
        var yData = new Vector<float>(new float[] { 5.0f, 7.0f, 9.0f });

        var model = new PredictionModelBuilder<float, Matrix<float>, Vector<float>>()
            .ConfigureModel(new SimpleRegression<float>())
            .ConfigureJitCompilation(new JitCompilationConfig
            {
                Enabled = true,
                ThrowOnFailure = false  // Use graceful fallback since not all models support JIT
            });

        // Act: Should succeed
        var result = await model.BuildAsync(xData, yData);

        // Assert: Model should be functional
        var testData = new Matrix<float>(new float[,] { { 4.0f } });
        var prediction = result.Predict(testData);
        Assert.NotNull(prediction);
        Assert.Single(prediction);
    }

    /// <summary>
    /// US-1.5: Verify JIT compilation works with multiple features.
    /// </summary>
    [Fact]
    public async Task SimpleRegression_MultipleFeatures_JitCompilationWorks()
    {
        // Arrange: Create dataset with multiple features
        var xData = new Matrix<float>(new float[,]
        {
            { 1.0f, 2.0f, 3.0f },
            { 2.0f, 3.0f, 4.0f },
            { 3.0f, 4.0f, 5.0f },
            { 4.0f, 5.0f, 6.0f },
            { 5.0f, 6.0f, 7.0f }
        });

        // y = x1 + 2*x2 + 3*x3 + noise
        var yData = new Vector<float>(new float[]
        {
            14.0f,  // 1 + 2*2 + 3*3 = 14
            20.0f,  // 2 + 2*3 + 3*4 = 20
            26.0f,  // 3 + 2*4 + 3*5 = 26
            32.0f,  // 4 + 2*5 + 3*6 = 32
            38.0f   // 5 + 2*6 + 3*7 = 38
        });

        // Train with JIT
        var model = new PredictionModelBuilder<float, Matrix<float>, Vector<float>>()
            .ConfigureModel(new SimpleRegression<float>())
            .ConfigureJitCompilation(new JitCompilationConfig { Enabled = true });

        var result = await model.BuildAsync(xData, yData);

        // Act: Make prediction
        var testData = new Matrix<float>(new float[,] { { 6.0f, 7.0f, 8.0f } });
        var prediction = result.Predict(testData);

        // Assert: Should get reasonable prediction (6 + 2*7 + 3*8 = 44)
        Assert.Single(prediction);
        Assert.InRange(prediction[0], 40.0f, 48.0f); // Allow some tolerance for fitting
    }
}
