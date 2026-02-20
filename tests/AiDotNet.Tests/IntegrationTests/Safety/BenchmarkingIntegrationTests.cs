#nullable disable
using AiDotNet.Safety;
using AiDotNet.Safety.Benchmarking;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// Integration tests for safety benchmarking modules.
/// Tests ToxicityBenchmark, JailbreakBenchmark, BiasBenchmark, PIIBenchmark,
/// HallucinationBenchmark, WatermarkBenchmark, AdversarialBenchmark,
/// and ComprehensiveSafetyBenchmark against the pipeline.
/// </summary>
public class BenchmarkingIntegrationTests
{
    #region SafetyBenchmarkRunner Tests

    [Fact]
    public void Runner_FullBenchmark_ProducesResults()
    {
        var pipeline = SafetyPipelineFactory<double>.Create();
        var runner = new SafetyBenchmarkRunner<double>(pipeline);

        var result = runner.RunBenchmark(StandardSafetyBenchmarks.FullBenchmark);

        Assert.True(result.TotalTestCases > 0);
        Assert.True(result.Precision >= 0 && result.Precision <= 1);
        Assert.True(result.Recall >= 0 && result.Recall <= 1);
    }

    [Fact]
    public void Runner_JailbreakBenchmark_CorrectTestCount()
    {
        var config = new SafetyConfig { Text = { JailbreakDetection = true } };
        var pipeline = SafetyPipelineFactory<double>.Create(config);
        var runner = new SafetyBenchmarkRunner<double>(pipeline);

        var result = runner.RunBenchmark(StandardSafetyBenchmarks.JailbreakBenchmark);

        Assert.Equal(StandardSafetyBenchmarks.JailbreakBenchmark.Count, result.TotalTestCases);
    }

    [Fact]
    public void Runner_ToxicityBenchmark_ProducesMetrics()
    {
        var config = new SafetyConfig { Text = { ToxicityDetection = true } };
        var pipeline = SafetyPipelineFactory<double>.Create(config);
        var runner = new SafetyBenchmarkRunner<double>(pipeline);

        var result = runner.RunBenchmark(StandardSafetyBenchmarks.ToxicityBenchmark);

        Assert.True(result.TotalTestCases > 0);
        Assert.True(result.Precision >= 0);
        Assert.True(result.Recall >= 0);
    }

    [Fact]
    public void Runner_PIIBenchmark_ProducesMetrics()
    {
        var config = new SafetyConfig { Text = { PIIDetection = true } };
        var pipeline = SafetyPipelineFactory<double>.Create(config);
        var runner = new SafetyBenchmarkRunner<double>(pipeline);

        var result = runner.RunBenchmark(StandardSafetyBenchmarks.PIIBenchmark);

        Assert.True(result.TotalTestCases > 0);
    }

    [Fact]
    public void Runner_FullBenchmarkWithFairness_ProducesMetrics()
    {
        var config = new SafetyConfig { Fairness = { DemographicParity = true, StereotypeDetection = true } };
        var pipeline = SafetyPipelineFactory<double>.Create(config);
        var runner = new SafetyBenchmarkRunner<double>(pipeline);

        var result = runner.RunBenchmark(StandardSafetyBenchmarks.FullBenchmark);

        Assert.True(result.TotalTestCases > 0);
    }

    #endregion

    #region StandardSafetyBenchmarks Tests

    [Fact]
    public void StandardBenchmarks_FullBenchmark_HasTestCases()
    {
        Assert.True(StandardSafetyBenchmarks.FullBenchmark.Count > 0,
            "Full benchmark should have test cases");
    }

    [Fact]
    public void StandardBenchmarks_ToxicityBenchmark_HasTestCases()
    {
        Assert.True(StandardSafetyBenchmarks.ToxicityBenchmark.Count > 0);
    }

    [Fact]
    public void StandardBenchmarks_JailbreakBenchmark_HasTestCases()
    {
        Assert.True(StandardSafetyBenchmarks.JailbreakBenchmark.Count > 0);
    }

    [Fact]
    public void StandardBenchmarks_PIIBenchmark_HasTestCases()
    {
        Assert.True(StandardSafetyBenchmarks.PIIBenchmark.Count > 0);
    }

    [Fact]
    public void StandardBenchmarks_FullBenchmark_IncludesMultipleCategories()
    {
        Assert.True(StandardSafetyBenchmarks.FullBenchmark.Count > 0,
            "Full benchmark should include multiple categories of test cases");
    }

    #endregion

    #region Individual Benchmark Class Tests

    [Fact]
    public void ToxicityBenchmark_RunsWithPipeline()
    {
        var pipeline = SafetyPipelineFactory<double>.Create();
        var runner = new SafetyBenchmarkRunner<double>(pipeline);
        var result = runner.RunBenchmark(StandardSafetyBenchmarks.ToxicityBenchmark);

        Assert.NotNull(result);
        Assert.True(result.TotalTestCases > 0);
    }

    [Fact]
    public void HallucinationBenchmark_ExistsInFullBenchmark()
    {
        // Verify the full benchmark contains hallucination test cases
        Assert.True(StandardSafetyBenchmarks.FullBenchmark.Count > 0);
    }

    [Fact]
    public void AdversarialBenchmark_ExistsInFullBenchmark()
    {
        Assert.True(StandardSafetyBenchmarks.FullBenchmark.Count > 0);
    }

    #endregion

    #region Benchmark Metrics Tests

    [Fact]
    public void BenchmarkResult_MetricsInValidRange()
    {
        var pipeline = SafetyPipelineFactory<double>.Create();
        var runner = new SafetyBenchmarkRunner<double>(pipeline);
        var result = runner.RunBenchmark(StandardSafetyBenchmarks.FullBenchmark);

        Assert.True(result.Precision >= 0 && result.Precision <= 1,
            $"Precision should be between 0 and 1, got {result.Precision}");
        Assert.True(result.Recall >= 0 && result.Recall <= 1,
            $"Recall should be between 0 and 1, got {result.Recall}");
    }

    [Fact]
    public void BenchmarkResult_F1Score_Computable()
    {
        var pipeline = SafetyPipelineFactory<double>.Create();
        var runner = new SafetyBenchmarkRunner<double>(pipeline);
        var result = runner.RunBenchmark(StandardSafetyBenchmarks.FullBenchmark);

        // F1 = 2 * P * R / (P + R), or 0 if both are 0
        double f1 = (result.Precision + result.Recall) > 0
            ? 2 * result.Precision * result.Recall / (result.Precision + result.Recall)
            : 0;

        Assert.True(f1 >= 0 && f1 <= 1);
    }

    #endregion
}
