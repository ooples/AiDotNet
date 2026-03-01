using AiDotNet.Benchmarking;
using AiDotNet.Benchmarking.Models;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Benchmarking;

/// <summary>
/// Deep integration tests for BenchmarkSuiteRegistry (descriptor lookups,
/// display names, suite kinds, factory methods) and BenchmarkReport
/// (OverallStatus logic, duration computation, suite composition).
/// </summary>
public class BenchmarkingDeepMathIntegrationTests
{
    // ============================
    // BenchmarkSuiteRegistry: GetAvailableSuites
    // ============================

    [Fact]
    public void GetAvailableSuites_ReturnsNonEmpty()
    {
        var suites = BenchmarkSuiteRegistry.GetAvailableSuites();
        Assert.NotEmpty(suites);
    }

    [Fact]
    public void GetAvailableSuites_Contains23Suites()
    {
        // Registry has 23 defined descriptors
        var suites = BenchmarkSuiteRegistry.GetAvailableSuites();
        Assert.Equal(23, suites.Count);
    }

    [Fact]
    public void GetAvailableSuites_SortedByKindThenDisplayName()
    {
        var suites = BenchmarkSuiteRegistry.GetAvailableSuites();

        for (int i = 1; i < suites.Count; i++)
        {
            var prev = suites[i - 1];
            var curr = suites[i];

            if (prev.Kind == curr.Kind)
            {
                // Within same kind, sorted by DisplayName
                Assert.True(string.Compare(prev.DisplayName, curr.DisplayName, StringComparison.Ordinal) <= 0,
                    $"Expected {prev.DisplayName} <= {curr.DisplayName} within kind {prev.Kind}");
            }
            else
            {
                // Different kinds are sorted by Kind enum value
                Assert.True(prev.Kind <= curr.Kind,
                    $"Expected kind {prev.Kind} <= {curr.Kind}");
            }
        }
    }

    [Fact]
    public void GetAvailableSuites_AllHaveNonEmptyDisplayName()
    {
        var suites = BenchmarkSuiteRegistry.GetAvailableSuites();

        foreach (var suite in suites)
        {
            Assert.False(string.IsNullOrWhiteSpace(suite.DisplayName),
                $"Suite {suite.Suite} has empty display name");
        }
    }

    [Fact]
    public void GetAvailableSuites_AllHaveNonEmptyDescription()
    {
        var suites = BenchmarkSuiteRegistry.GetAvailableSuites();

        foreach (var suite in suites)
        {
            Assert.False(string.IsNullOrWhiteSpace(suite.Description),
                $"Suite {suite.Suite} has empty description");
        }
    }

    // ============================
    // BenchmarkSuiteRegistry: GetSuiteKind
    // ============================

    [Theory]
    [InlineData(BenchmarkSuite.LEAF, BenchmarkSuiteKind.DatasetSuite)]
    [InlineData(BenchmarkSuite.FEMNIST, BenchmarkSuiteKind.DatasetSuite)]
    [InlineData(BenchmarkSuite.Sent140, BenchmarkSuiteKind.DatasetSuite)]
    [InlineData(BenchmarkSuite.Shakespeare, BenchmarkSuiteKind.DatasetSuite)]
    [InlineData(BenchmarkSuite.Reddit, BenchmarkSuiteKind.DatasetSuite)]
    [InlineData(BenchmarkSuite.StackOverflow, BenchmarkSuiteKind.DatasetSuite)]
    [InlineData(BenchmarkSuite.CIFAR10, BenchmarkSuiteKind.DatasetSuite)]
    [InlineData(BenchmarkSuite.CIFAR100, BenchmarkSuiteKind.DatasetSuite)]
    [InlineData(BenchmarkSuite.TabularNonIID, BenchmarkSuiteKind.DatasetSuite)]
    public void GetSuiteKind_DatasetSuites_ReturnDatasetSuite(BenchmarkSuite suite, BenchmarkSuiteKind expectedKind)
    {
        Assert.Equal(expectedKind, BenchmarkSuiteRegistry.GetSuiteKind(suite));
    }

    [Theory]
    [InlineData(BenchmarkSuite.GSM8K, BenchmarkSuiteKind.Reasoning)]
    [InlineData(BenchmarkSuite.MATH, BenchmarkSuiteKind.Reasoning)]
    [InlineData(BenchmarkSuite.MMLU, BenchmarkSuiteKind.Reasoning)]
    [InlineData(BenchmarkSuite.TruthfulQA, BenchmarkSuiteKind.Reasoning)]
    [InlineData(BenchmarkSuite.ARCAGI, BenchmarkSuiteKind.Reasoning)]
    [InlineData(BenchmarkSuite.DROP, BenchmarkSuiteKind.Reasoning)]
    [InlineData(BenchmarkSuite.BoolQ, BenchmarkSuiteKind.Reasoning)]
    [InlineData(BenchmarkSuite.PIQA, BenchmarkSuiteKind.Reasoning)]
    [InlineData(BenchmarkSuite.CommonsenseQA, BenchmarkSuiteKind.Reasoning)]
    [InlineData(BenchmarkSuite.WinoGrande, BenchmarkSuiteKind.Reasoning)]
    [InlineData(BenchmarkSuite.HellaSwag, BenchmarkSuiteKind.Reasoning)]
    [InlineData(BenchmarkSuite.HumanEval, BenchmarkSuiteKind.Reasoning)]
    [InlineData(BenchmarkSuite.MBPP, BenchmarkSuiteKind.Reasoning)]
    [InlineData(BenchmarkSuite.LogiQA, BenchmarkSuiteKind.Reasoning)]
    public void GetSuiteKind_ReasoningSuites_ReturnReasoning(BenchmarkSuite suite, BenchmarkSuiteKind expectedKind)
    {
        Assert.Equal(expectedKind, BenchmarkSuiteRegistry.GetSuiteKind(suite));
    }

    [Fact]
    public void GetSuiteKind_9DatasetSuites_14ReasoningSuites()
    {
        var suites = BenchmarkSuiteRegistry.GetAvailableSuites();
        int datasetCount = suites.Count(s => s.Kind == BenchmarkSuiteKind.DatasetSuite);
        int reasoningCount = suites.Count(s => s.Kind == BenchmarkSuiteKind.Reasoning);

        Assert.Equal(9, datasetCount);
        Assert.Equal(14, reasoningCount);
        Assert.Equal(23, datasetCount + reasoningCount);
    }

    // ============================
    // BenchmarkSuiteRegistry: GetDisplayName
    // ============================

    [Theory]
    [InlineData(BenchmarkSuite.LEAF, "LEAF")]
    [InlineData(BenchmarkSuite.FEMNIST, "FEMNIST")]
    [InlineData(BenchmarkSuite.Sent140, "Sent140")]
    [InlineData(BenchmarkSuite.GSM8K, "GSM8K")]
    [InlineData(BenchmarkSuite.MATH, "MATH")]
    [InlineData(BenchmarkSuite.MMLU, "MMLU")]
    [InlineData(BenchmarkSuite.TruthfulQA, "TruthfulQA")]
    [InlineData(BenchmarkSuite.ARCAGI, "ARC-AGI")]
    [InlineData(BenchmarkSuite.CIFAR10, "CIFAR-10")]
    [InlineData(BenchmarkSuite.CIFAR100, "CIFAR-100")]
    [InlineData(BenchmarkSuite.HumanEval, "HumanEval")]
    [InlineData(BenchmarkSuite.MBPP, "MBPP")]
    public void GetDisplayName_KnownSuites_CorrectNames(BenchmarkSuite suite, string expectedName)
    {
        Assert.Equal(expectedName, BenchmarkSuiteRegistry.GetDisplayName(suite));
    }

    // ============================
    // BenchmarkSuiteRegistry: CreateReasoningBenchmark
    // ============================

    [Theory]
    [InlineData(BenchmarkSuite.GSM8K)]
    [InlineData(BenchmarkSuite.MATH)]
    [InlineData(BenchmarkSuite.MMLU)]
    [InlineData(BenchmarkSuite.TruthfulQA)]
    [InlineData(BenchmarkSuite.ARCAGI)]
    [InlineData(BenchmarkSuite.DROP)]
    [InlineData(BenchmarkSuite.BoolQ)]
    [InlineData(BenchmarkSuite.PIQA)]
    [InlineData(BenchmarkSuite.CommonsenseQA)]
    [InlineData(BenchmarkSuite.WinoGrande)]
    [InlineData(BenchmarkSuite.HellaSwag)]
    [InlineData(BenchmarkSuite.HumanEval)]
    [InlineData(BenchmarkSuite.MBPP)]
    [InlineData(BenchmarkSuite.LogiQA)]
    public void CreateReasoningBenchmark_AllReasoningSuites_ReturnNonNull(BenchmarkSuite suite)
    {
        var benchmark = BenchmarkSuiteRegistry.CreateReasoningBenchmark(suite);
        Assert.NotNull(benchmark);
    }

    [Fact]
    public void CreateReasoningBenchmark_DatasetSuite_ThrowsNotSupported()
    {
        Assert.Throws<NotSupportedException>(() =>
            BenchmarkSuiteRegistry.CreateReasoningBenchmark(BenchmarkSuite.LEAF));
    }

    [Fact]
    public void CreateReasoningBenchmark_MultipleCalls_ReturnDistinctInstances()
    {
        var b1 = BenchmarkSuiteRegistry.CreateReasoningBenchmark(BenchmarkSuite.GSM8K);
        var b2 = BenchmarkSuiteRegistry.CreateReasoningBenchmark(BenchmarkSuite.GSM8K);

        // Factory creates new instances each time
        Assert.NotSame(b1, b2);
    }

    // ============================
    // BenchmarkReport: OverallStatus Logic
    // ============================

    [Fact]
    public void OverallStatus_EmptySuites_Skipped()
    {
        var report = new BenchmarkReport();
        Assert.Equal(BenchmarkExecutionStatus.Skipped, report.OverallStatus);
    }

    [Fact]
    public void OverallStatus_AllSucceeded_Succeeded()
    {
        var report = new BenchmarkReport
        {
            Suites = new[]
            {
                new BenchmarkSuiteReport { Status = BenchmarkExecutionStatus.Succeeded },
                new BenchmarkSuiteReport { Status = BenchmarkExecutionStatus.Succeeded }
            }
        };

        Assert.Equal(BenchmarkExecutionStatus.Succeeded, report.OverallStatus);
    }

    [Fact]
    public void OverallStatus_AnyFailed_Failed()
    {
        var report = new BenchmarkReport
        {
            Suites = new[]
            {
                new BenchmarkSuiteReport { Status = BenchmarkExecutionStatus.Succeeded },
                new BenchmarkSuiteReport { Status = BenchmarkExecutionStatus.Failed },
                new BenchmarkSuiteReport { Status = BenchmarkExecutionStatus.Succeeded }
            }
        };

        Assert.Equal(BenchmarkExecutionStatus.Failed, report.OverallStatus);
    }

    [Fact]
    public void OverallStatus_AllSkipped_Skipped()
    {
        var report = new BenchmarkReport
        {
            Suites = new[]
            {
                new BenchmarkSuiteReport { Status = BenchmarkExecutionStatus.Skipped },
                new BenchmarkSuiteReport { Status = BenchmarkExecutionStatus.Skipped }
            }
        };

        Assert.Equal(BenchmarkExecutionStatus.Skipped, report.OverallStatus);
    }

    [Fact]
    public void OverallStatus_MixSucceededAndSkipped_Succeeded()
    {
        // Not all skipped + no failures = succeeded
        var report = new BenchmarkReport
        {
            Suites = new[]
            {
                new BenchmarkSuiteReport { Status = BenchmarkExecutionStatus.Succeeded },
                new BenchmarkSuiteReport { Status = BenchmarkExecutionStatus.Skipped }
            }
        };

        Assert.Equal(BenchmarkExecutionStatus.Succeeded, report.OverallStatus);
    }

    [Fact]
    public void OverallStatus_FailedTakesPriority_OverSucceeded()
    {
        // Failed takes priority over all other statuses
        var report = new BenchmarkReport
        {
            Suites = new[]
            {
                new BenchmarkSuiteReport { Status = BenchmarkExecutionStatus.Succeeded },
                new BenchmarkSuiteReport { Status = BenchmarkExecutionStatus.Failed },
                new BenchmarkSuiteReport { Status = BenchmarkExecutionStatus.Skipped }
            }
        };

        Assert.Equal(BenchmarkExecutionStatus.Failed, report.OverallStatus);
    }

    [Fact]
    public void OverallStatus_SingleFailed_Failed()
    {
        var report = new BenchmarkReport
        {
            Suites = new[]
            {
                new BenchmarkSuiteReport { Status = BenchmarkExecutionStatus.Failed }
            }
        };

        Assert.Equal(BenchmarkExecutionStatus.Failed, report.OverallStatus);
    }

    [Fact]
    public void OverallStatus_SingleSucceeded_Succeeded()
    {
        var report = new BenchmarkReport
        {
            Suites = new[]
            {
                new BenchmarkSuiteReport { Status = BenchmarkExecutionStatus.Succeeded }
            }
        };

        Assert.Equal(BenchmarkExecutionStatus.Succeeded, report.OverallStatus);
    }

    // ============================
    // BenchmarkReport: Duration Computation
    // ============================

    [Fact]
    public void TotalDuration_ComputedFromStartAndEnd()
    {
        var start = new DateTimeOffset(2025, 1, 1, 0, 0, 0, TimeSpan.Zero);
        var end = new DateTimeOffset(2025, 1, 1, 1, 30, 0, TimeSpan.Zero);

        var report = new BenchmarkReport
        {
            StartedUtc = start,
            EndedUtc = end
        };

        Assert.Equal(TimeSpan.FromMinutes(90), report.TotalDuration);
    }

    [Fact]
    public void BenchmarkSuiteReport_Duration_ComputedFromStartAndEnd()
    {
        var start = new DateTimeOffset(2025, 1, 1, 0, 0, 0, TimeSpan.Zero);
        var end = new DateTimeOffset(2025, 1, 1, 0, 5, 30, TimeSpan.Zero);

        var suiteReport = new BenchmarkSuiteReport
        {
            StartedUtc = start,
            EndedUtc = end
        };

        Assert.Equal(TimeSpan.FromSeconds(330), suiteReport.Duration);
    }

    // ============================
    // BenchmarkSuiteDescriptor Tests
    // ============================

    [Fact]
    public void BenchmarkSuiteDescriptor_AllSuitesHaveMatchingSuiteProperty()
    {
        var suites = BenchmarkSuiteRegistry.GetAvailableSuites();

        foreach (var descriptor in suites)
        {
            // Each descriptor's Suite property should match what GetSuiteKind returns
            Assert.Equal(descriptor.Kind, BenchmarkSuiteRegistry.GetSuiteKind(descriptor.Suite));

            // Each descriptor's DisplayName should match what GetDisplayName returns
            Assert.Equal(descriptor.DisplayName, BenchmarkSuiteRegistry.GetDisplayName(descriptor.Suite));
        }
    }

    // ============================
    // BenchmarkMetricValue Tests
    // ============================

    [Fact]
    public void BenchmarkMetricValue_DefaultValue_Zero()
    {
        var metric = new BenchmarkMetricValue();
        Assert.Equal(0.0, metric.Value);
    }

    // ============================
    // BenchmarkCategoryResult Tests
    // ============================

    [Fact]
    public void BenchmarkCategoryResult_DefaultAccuracy_Zero()
    {
        var result = new BenchmarkCategoryResult();
        Assert.Equal(0.0, result.Accuracy);
        Assert.Equal(string.Empty, result.Category);
    }
}
