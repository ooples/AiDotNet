using AiDotNet.Benchmarking;
using AiDotNet.Benchmarking.Models;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Benchmarking;

/// <summary>
/// Integration tests for the Benchmarking module.
/// Tests BenchmarkSuiteRegistry, BenchmarkReport, and related model classes.
/// </summary>
public class BenchmarkingIntegrationTests
{
    #region BenchmarkSuiteRegistry Tests

    [Fact]
    public void BenchmarkSuiteRegistry_GetAvailableSuites_ReturnsAllRegisteredSuites()
    {
        // Act
        var suites = BenchmarkSuiteRegistry.GetAvailableSuites();

        // Assert
        Assert.NotNull(suites);
        Assert.NotEmpty(suites);

        // Should contain both Reasoning and DatasetSuite kinds
        Assert.Contains(suites, s => s.Kind == BenchmarkSuiteKind.Reasoning);
        Assert.Contains(suites, s => s.Kind == BenchmarkSuiteKind.DatasetSuite);

        // Should contain specific known suites
        Assert.Contains(suites, s => s.Suite == BenchmarkSuite.GSM8K);
        Assert.Contains(suites, s => s.Suite == BenchmarkSuite.MMLU);
        Assert.Contains(suites, s => s.Suite == BenchmarkSuite.LEAF);
    }

    [Fact]
    public void BenchmarkSuiteRegistry_GetAvailableSuites_SuitesAreOrderedByKindThenName()
    {
        // Act
        var suites = BenchmarkSuiteRegistry.GetAvailableSuites();

        // Assert - verify ordering (by Kind, then by DisplayName)
        BenchmarkSuiteKind? previousKind = null;
        string? previousName = null;

        foreach (var suite in suites)
        {
            if (previousKind.HasValue)
            {
                // Either the kind increased, or it's the same kind with alphabetically greater/equal name
                bool kindIncreased = suite.Kind > previousKind.Value;
                bool sameKindAlphabetical = suite.Kind == previousKind.Value &&
                    string.Compare(suite.DisplayName, previousName, StringComparison.Ordinal) >= 0;

                Assert.True(kindIncreased || sameKindAlphabetical,
                    $"Suites should be ordered by Kind then DisplayName. Found {suite.DisplayName} after {previousName}");
            }

            previousKind = suite.Kind;
            previousName = suite.DisplayName;
        }
    }

    [Fact]
    public void BenchmarkSuiteRegistry_GetAvailableSuites_AllDescriptorsHaveRequiredFields()
    {
        // Act
        var suites = BenchmarkSuiteRegistry.GetAvailableSuites();

        // Assert
        foreach (var descriptor in suites)
        {
            Assert.NotNull(descriptor.DisplayName);
            Assert.NotEmpty(descriptor.DisplayName);
            Assert.NotNull(descriptor.Description);
            Assert.NotEmpty(descriptor.Description);
        }
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
    [InlineData(BenchmarkSuite.LEAF, BenchmarkSuiteKind.DatasetSuite)]
    [InlineData(BenchmarkSuite.FEMNIST, BenchmarkSuiteKind.DatasetSuite)]
    [InlineData(BenchmarkSuite.Sent140, BenchmarkSuiteKind.DatasetSuite)]
    [InlineData(BenchmarkSuite.Shakespeare, BenchmarkSuiteKind.DatasetSuite)]
    [InlineData(BenchmarkSuite.Reddit, BenchmarkSuiteKind.DatasetSuite)]
    [InlineData(BenchmarkSuite.StackOverflow, BenchmarkSuiteKind.DatasetSuite)]
    [InlineData(BenchmarkSuite.CIFAR10, BenchmarkSuiteKind.DatasetSuite)]
    [InlineData(BenchmarkSuite.CIFAR100, BenchmarkSuiteKind.DatasetSuite)]
    [InlineData(BenchmarkSuite.TabularNonIID, BenchmarkSuiteKind.DatasetSuite)]
    public void BenchmarkSuiteRegistry_GetSuiteKind_ReturnsCorrectKind(BenchmarkSuite suite, BenchmarkSuiteKind expectedKind)
    {
        // Act
        var kind = BenchmarkSuiteRegistry.GetSuiteKind(suite);

        // Assert
        Assert.Equal(expectedKind, kind);
    }

    [Theory]
    [InlineData(BenchmarkSuite.GSM8K, "GSM8K")]
    [InlineData(BenchmarkSuite.MATH, "MATH")]
    [InlineData(BenchmarkSuite.MMLU, "MMLU")]
    [InlineData(BenchmarkSuite.TruthfulQA, "TruthfulQA")]
    [InlineData(BenchmarkSuite.ARCAGI, "ARC-AGI")]
    [InlineData(BenchmarkSuite.DROP, "DROP")]
    [InlineData(BenchmarkSuite.BoolQ, "BoolQ")]
    [InlineData(BenchmarkSuite.PIQA, "PIQA")]
    [InlineData(BenchmarkSuite.CommonsenseQA, "CommonsenseQA")]
    [InlineData(BenchmarkSuite.WinoGrande, "WinoGrande")]
    [InlineData(BenchmarkSuite.HellaSwag, "HellaSwag")]
    [InlineData(BenchmarkSuite.HumanEval, "HumanEval")]
    [InlineData(BenchmarkSuite.MBPP, "MBPP")]
    [InlineData(BenchmarkSuite.LogiQA, "LogiQA")]
    [InlineData(BenchmarkSuite.LEAF, "LEAF")]
    [InlineData(BenchmarkSuite.FEMNIST, "FEMNIST")]
    [InlineData(BenchmarkSuite.Sent140, "Sent140")]
    [InlineData(BenchmarkSuite.Shakespeare, "Shakespeare")]
    [InlineData(BenchmarkSuite.Reddit, "Reddit")]
    [InlineData(BenchmarkSuite.StackOverflow, "StackOverflow")]
    [InlineData(BenchmarkSuite.CIFAR10, "CIFAR-10")]
    [InlineData(BenchmarkSuite.CIFAR100, "CIFAR-100")]
    [InlineData(BenchmarkSuite.TabularNonIID, "Tabular (Non-IID)")]
    public void BenchmarkSuiteRegistry_GetDisplayName_ReturnsCorrectName(BenchmarkSuite suite, string expectedName)
    {
        // Act
        var name = BenchmarkSuiteRegistry.GetDisplayName(suite);

        // Assert
        Assert.Equal(expectedName, name);
    }

    [Fact]
    public void BenchmarkSuiteRegistry_GetSuiteKind_WithInvalidSuite_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var invalidSuite = (BenchmarkSuite)9999;

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => BenchmarkSuiteRegistry.GetSuiteKind(invalidSuite));
    }

    [Fact]
    public void BenchmarkSuiteRegistry_GetDisplayName_WithInvalidSuite_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var invalidSuite = (BenchmarkSuite)9999;

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => BenchmarkSuiteRegistry.GetDisplayName(invalidSuite));
    }

    #endregion

    #region BenchmarkReport Tests

    [Fact]
    public void BenchmarkReport_TotalDuration_CalculatesCorrectly()
    {
        // Arrange
        var startTime = DateTimeOffset.UtcNow;
        var endTime = startTime.AddMinutes(5);

        var report = new BenchmarkReport();
        // Use reflection to set internal properties
        SetInternalProperty(report, nameof(BenchmarkReport.StartedUtc), startTime);
        SetInternalProperty(report, nameof(BenchmarkReport.EndedUtc), endTime);

        // Act
        var duration = report.TotalDuration;

        // Assert
        Assert.Equal(TimeSpan.FromMinutes(5), duration);
    }

    [Fact]
    public void BenchmarkReport_OverallStatus_WithNoSuites_ReturnsSkipped()
    {
        // Arrange
        var report = new BenchmarkReport();
        SetInternalProperty(report, nameof(BenchmarkReport.Suites), Array.Empty<BenchmarkSuiteReport>());

        // Act
        var status = report.OverallStatus;

        // Assert
        Assert.Equal(BenchmarkExecutionStatus.Skipped, status);
    }

    [Fact]
    public void BenchmarkReport_OverallStatus_WithAllSucceeded_ReturnsSucceeded()
    {
        // Arrange
        var suiteReports = new[]
        {
            CreateSuiteReport(BenchmarkSuite.GSM8K, BenchmarkExecutionStatus.Succeeded),
            CreateSuiteReport(BenchmarkSuite.MMLU, BenchmarkExecutionStatus.Succeeded)
        };

        var report = new BenchmarkReport();
        SetInternalProperty(report, nameof(BenchmarkReport.Suites), (IReadOnlyList<BenchmarkSuiteReport>)suiteReports);

        // Act
        var status = report.OverallStatus;

        // Assert
        Assert.Equal(BenchmarkExecutionStatus.Succeeded, status);
    }

    [Fact]
    public void BenchmarkReport_OverallStatus_WithOneFailed_ReturnsFailed()
    {
        // Arrange
        var suiteReports = new[]
        {
            CreateSuiteReport(BenchmarkSuite.GSM8K, BenchmarkExecutionStatus.Succeeded),
            CreateSuiteReport(BenchmarkSuite.MMLU, BenchmarkExecutionStatus.Failed)
        };

        var report = new BenchmarkReport();
        SetInternalProperty(report, nameof(BenchmarkReport.Suites), (IReadOnlyList<BenchmarkSuiteReport>)suiteReports);

        // Act
        var status = report.OverallStatus;

        // Assert
        Assert.Equal(BenchmarkExecutionStatus.Failed, status);
    }

    [Fact]
    public void BenchmarkReport_OverallStatus_WithAllSkipped_ReturnsSkipped()
    {
        // Arrange
        var suiteReports = new[]
        {
            CreateSuiteReport(BenchmarkSuite.GSM8K, BenchmarkExecutionStatus.Skipped),
            CreateSuiteReport(BenchmarkSuite.MMLU, BenchmarkExecutionStatus.Skipped)
        };

        var report = new BenchmarkReport();
        SetInternalProperty(report, nameof(BenchmarkReport.Suites), (IReadOnlyList<BenchmarkSuiteReport>)suiteReports);

        // Act
        var status = report.OverallStatus;

        // Assert
        Assert.Equal(BenchmarkExecutionStatus.Skipped, status);
    }

    [Fact]
    public void BenchmarkReport_OverallStatus_WithMixedSuccessAndSkipped_ReturnsSucceeded()
    {
        // Arrange
        var suiteReports = new[]
        {
            CreateSuiteReport(BenchmarkSuite.GSM8K, BenchmarkExecutionStatus.Succeeded),
            CreateSuiteReport(BenchmarkSuite.MMLU, BenchmarkExecutionStatus.Skipped)
        };

        var report = new BenchmarkReport();
        SetInternalProperty(report, nameof(BenchmarkReport.Suites), (IReadOnlyList<BenchmarkSuiteReport>)suiteReports);

        // Act
        var status = report.OverallStatus;

        // Assert
        Assert.Equal(BenchmarkExecutionStatus.Succeeded, status);
    }

    #endregion

    #region BenchmarkSuiteReport Tests

    [Fact]
    public void BenchmarkSuiteReport_Duration_CalculatesCorrectly()
    {
        // Arrange
        var startTime = DateTimeOffset.UtcNow;
        var endTime = startTime.AddSeconds(30);

        var report = new BenchmarkSuiteReport();
        SetInternalProperty(report, nameof(BenchmarkSuiteReport.StartedUtc), startTime);
        SetInternalProperty(report, nameof(BenchmarkSuiteReport.EndedUtc), endTime);

        // Act
        var duration = report.Duration;

        // Assert
        Assert.Equal(TimeSpan.FromSeconds(30), duration);
    }

    [Fact]
    public void BenchmarkSuiteReport_DefaultValues_AreCorrect()
    {
        // Act
        var report = new BenchmarkSuiteReport();

        // Assert
        Assert.Equal(string.Empty, report.Name);
        Assert.Null(report.FailureReason);
        Assert.Empty(report.Metrics);
        Assert.Null(report.CategoryAccuracies);
        Assert.Null(report.DataSelection);
    }

    [Fact]
    public void BenchmarkSuiteReport_WithMetrics_StoresMetricsCorrectly()
    {
        // Arrange
        var metrics = new[]
        {
            new BenchmarkMetricValue { Metric = BenchmarkMetric.Accuracy, Value = 0.85 },
            new BenchmarkMetricValue { Metric = BenchmarkMetric.TotalEvaluated, Value = 100 }
        };

        var report = new BenchmarkSuiteReport();
        SetInternalProperty(report, nameof(BenchmarkSuiteReport.Metrics), (IReadOnlyList<BenchmarkMetricValue>)metrics);

        // Assert
        Assert.Equal(2, report.Metrics.Count);
        Assert.Contains(report.Metrics, m => m.Metric == BenchmarkMetric.Accuracy && Math.Abs(m.Value - 0.85) < 0.001);
        Assert.Contains(report.Metrics, m => m.Metric == BenchmarkMetric.TotalEvaluated && Math.Abs(m.Value - 100) < 0.001);
    }

    [Fact]
    public void BenchmarkSuiteReport_WithCategoryAccuracies_StoresCategoriesCorrectly()
    {
        // Arrange
        var categories = new[]
        {
            new BenchmarkCategoryResult { Category = "Math", Accuracy = 0.9 },
            new BenchmarkCategoryResult { Category = "Science", Accuracy = 0.8 }
        };

        var report = new BenchmarkSuiteReport();
        SetInternalProperty(report, nameof(BenchmarkSuiteReport.CategoryAccuracies), (IReadOnlyList<BenchmarkCategoryResult>)categories);

        // Assert
        Assert.NotNull(report.CategoryAccuracies);
        Assert.Equal(2, report.CategoryAccuracies.Count);
    }

    [Fact]
    public void BenchmarkSuiteReport_WithDataSelection_StoresDataSelectionCorrectly()
    {
        // Arrange
        var dataSelection = new BenchmarkDataSelectionSummary
        {
            ClientsUsed = 10,
            TrainSamplesUsed = 1000,
            TestSamplesUsed = 200,
            FeatureCount = 50,
            CiMode = true,
            Seed = 42,
            MaxSamplesPerUser = 100
        };

        var report = new BenchmarkSuiteReport();
        SetInternalProperty(report, nameof(BenchmarkSuiteReport.DataSelection), dataSelection);

        // Assert
        Assert.NotNull(report.DataSelection);
        Assert.Equal(10, report.DataSelection.ClientsUsed);
        Assert.Equal(1000, report.DataSelection.TrainSamplesUsed);
        Assert.Equal(200, report.DataSelection.TestSamplesUsed);
        Assert.Equal(50, report.DataSelection.FeatureCount);
        Assert.True(report.DataSelection.CiMode);
        Assert.Equal(42, report.DataSelection.Seed);
        Assert.Equal(100, report.DataSelection.MaxSamplesPerUser);
    }

    #endregion

    #region BenchmarkMetricValue Tests

    [Theory]
    [InlineData(BenchmarkMetric.Accuracy, 0.95)]
    [InlineData(BenchmarkMetric.AverageConfidence, 0.8)]
    [InlineData(BenchmarkMetric.TotalEvaluated, 1000)]
    [InlineData(BenchmarkMetric.CorrectCount, 950)]
    [InlineData(BenchmarkMetric.TotalDurationMilliseconds, 5000.5)]
    [InlineData(BenchmarkMetric.AverageTimePerItemMilliseconds, 5.0)]
    [InlineData(BenchmarkMetric.MeanSquaredError, 0.01)]
    [InlineData(BenchmarkMetric.RootMeanSquaredError, 0.1)]
    public void BenchmarkMetricValue_StoresMetricAndValue(BenchmarkMetric metric, double value)
    {
        // Arrange & Act
        var metricValue = new BenchmarkMetricValue { Metric = metric, Value = value };

        // Assert
        Assert.Equal(metric, metricValue.Metric);
        Assert.Equal(value, metricValue.Value, precision: 5);
    }

    #endregion

    #region BenchmarkSuiteDescriptor Tests

    [Fact]
    public void BenchmarkSuiteDescriptor_DefaultValues_AreCorrect()
    {
        // Act
        var descriptor = new BenchmarkSuiteDescriptor();

        // Assert
        Assert.Equal(string.Empty, descriptor.DisplayName);
        Assert.Equal(string.Empty, descriptor.Description);
    }

    [Fact]
    public void BenchmarkSuiteDescriptor_AllDescriptorsInRegistry_HaveValidProperties()
    {
        // Act
        var suites = BenchmarkSuiteRegistry.GetAvailableSuites();

        // Assert
        foreach (var descriptor in suites)
        {
            // DisplayName should not be empty
            Assert.False(string.IsNullOrWhiteSpace(descriptor.DisplayName),
                $"Suite {descriptor.Suite} has empty DisplayName");

            // Description should not be empty
            Assert.False(string.IsNullOrWhiteSpace(descriptor.Description),
                $"Suite {descriptor.Suite} has empty Description");

            // Kind should be a valid enum value
            Assert.True(Enum.IsDefined(typeof(BenchmarkSuiteKind), descriptor.Kind),
                $"Suite {descriptor.Suite} has invalid Kind");
        }
    }

    #endregion

    #region BenchmarkCategoryResult Tests

    [Fact]
    public void BenchmarkCategoryResult_DefaultValues_AreCorrect()
    {
        // Act
        var result = new BenchmarkCategoryResult();

        // Assert
        Assert.Equal(string.Empty, result.Category);
        Assert.Equal(0.0, result.Accuracy);
    }

    [Theory]
    [InlineData("Mathematics", 0.95)]
    [InlineData("Physics", 0.88)]
    [InlineData("Computer Science", 0.92)]
    [InlineData("", 0.5)]
    public void BenchmarkCategoryResult_StoresCategoryAndAccuracy(string category, double accuracy)
    {
        // Arrange & Act
        var result = new BenchmarkCategoryResult { Category = category, Accuracy = accuracy };

        // Assert
        Assert.Equal(category, result.Category);
        Assert.Equal(accuracy, result.Accuracy, precision: 5);
    }

    [Fact]
    public void BenchmarkCategoryResult_AccuracyBoundaries_HandlesEdgeCases()
    {
        // Arrange & Act
        var resultZero = new BenchmarkCategoryResult { Category = "Test", Accuracy = 0.0 };
        var resultOne = new BenchmarkCategoryResult { Category = "Test", Accuracy = 1.0 };

        // Assert
        Assert.Equal(0.0, resultZero.Accuracy);
        Assert.Equal(1.0, resultOne.Accuracy);
    }

    #endregion

    #region BenchmarkDataSelectionSummary Tests

    [Fact]
    public void BenchmarkDataSelectionSummary_DefaultValues_AreCorrect()
    {
        // Act
        var summary = new BenchmarkDataSelectionSummary();

        // Assert
        Assert.Equal(0, summary.ClientsUsed);
        Assert.Equal(0, summary.TrainSamplesUsed);
        Assert.Equal(0, summary.TestSamplesUsed);
        Assert.Equal(0, summary.FeatureCount);
        Assert.False(summary.CiMode);
        Assert.Equal(0, summary.Seed);
        Assert.Equal(0, summary.MaxSamplesPerUser);
    }

    [Fact]
    public void BenchmarkDataSelectionSummary_AllProperties_CanBeSet()
    {
        // Arrange & Act
        var summary = new BenchmarkDataSelectionSummary
        {
            ClientsUsed = 100,
            TrainSamplesUsed = 50000,
            TestSamplesUsed = 10000,
            FeatureCount = 784,
            CiMode = true,
            Seed = 123456,
            MaxSamplesPerUser = 500
        };

        // Assert
        Assert.Equal(100, summary.ClientsUsed);
        Assert.Equal(50000, summary.TrainSamplesUsed);
        Assert.Equal(10000, summary.TestSamplesUsed);
        Assert.Equal(784, summary.FeatureCount);
        Assert.True(summary.CiMode);
        Assert.Equal(123456, summary.Seed);
        Assert.Equal(500, summary.MaxSamplesPerUser);
    }

    #endregion

    #region BenchmarkSuite Enum Coverage Tests

    [Fact]
    public void BenchmarkSuite_AllReasoningSuites_AreRegistered()
    {
        // Arrange
        var reasoningSuites = new[]
        {
            BenchmarkSuite.GSM8K,
            BenchmarkSuite.MATH,
            BenchmarkSuite.MMLU,
            BenchmarkSuite.TruthfulQA,
            BenchmarkSuite.ARCAGI,
            BenchmarkSuite.DROP,
            BenchmarkSuite.BoolQ,
            BenchmarkSuite.PIQA,
            BenchmarkSuite.CommonsenseQA,
            BenchmarkSuite.WinoGrande,
            BenchmarkSuite.HellaSwag,
            BenchmarkSuite.HumanEval,
            BenchmarkSuite.MBPP,
            BenchmarkSuite.LogiQA
        };

        // Act & Assert
        foreach (var suite in reasoningSuites)
        {
            var kind = BenchmarkSuiteRegistry.GetSuiteKind(suite);
            Assert.Equal(BenchmarkSuiteKind.Reasoning, kind);

            var name = BenchmarkSuiteRegistry.GetDisplayName(suite);
            Assert.NotNull(name);
            Assert.NotEmpty(name);
        }
    }

    [Fact]
    public void BenchmarkSuite_AllDatasetSuites_AreRegistered()
    {
        // Arrange
        var datasetSuites = new[]
        {
            BenchmarkSuite.LEAF,
            BenchmarkSuite.FEMNIST,
            BenchmarkSuite.Sent140,
            BenchmarkSuite.Shakespeare,
            BenchmarkSuite.Reddit,
            BenchmarkSuite.StackOverflow,
            BenchmarkSuite.CIFAR10,
            BenchmarkSuite.CIFAR100,
            BenchmarkSuite.TabularNonIID
        };

        // Act & Assert
        foreach (var suite in datasetSuites)
        {
            var kind = BenchmarkSuiteRegistry.GetSuiteKind(suite);
            Assert.Equal(BenchmarkSuiteKind.DatasetSuite, kind);

            var name = BenchmarkSuiteRegistry.GetDisplayName(suite);
            Assert.NotNull(name);
            Assert.NotEmpty(name);
        }
    }

    [Fact]
    public void BenchmarkSuite_TotalRegisteredCount_MatchesExpected()
    {
        // Act
        var suites = BenchmarkSuiteRegistry.GetAvailableSuites();

        // Assert - We expect 23 total suites (14 reasoning + 9 dataset)
        Assert.Equal(23, suites.Count);
    }

    #endregion

    #region BenchmarkMetric Enum Coverage Tests

    [Fact]
    public void BenchmarkMetric_AllValues_AreValid()
    {
        // Act
        var metrics = (BenchmarkMetric[])Enum.GetValues(typeof(BenchmarkMetric));

        // Assert
        Assert.Equal(8, metrics.Length);

        Assert.Contains(BenchmarkMetric.Accuracy, (IEnumerable<BenchmarkMetric>)metrics);
        Assert.Contains(BenchmarkMetric.AverageConfidence, (IEnumerable<BenchmarkMetric>)metrics);
        Assert.Contains(BenchmarkMetric.TotalEvaluated, (IEnumerable<BenchmarkMetric>)metrics);
        Assert.Contains(BenchmarkMetric.CorrectCount, (IEnumerable<BenchmarkMetric>)metrics);
        Assert.Contains(BenchmarkMetric.TotalDurationMilliseconds, (IEnumerable<BenchmarkMetric>)metrics);
        Assert.Contains(BenchmarkMetric.AverageTimePerItemMilliseconds, (IEnumerable<BenchmarkMetric>)metrics);
        Assert.Contains(BenchmarkMetric.MeanSquaredError, (IEnumerable<BenchmarkMetric>)metrics);
        Assert.Contains(BenchmarkMetric.RootMeanSquaredError, (IEnumerable<BenchmarkMetric>)metrics);
    }

    #endregion

    #region BenchmarkExecutionStatus Enum Tests

    [Fact]
    public void BenchmarkExecutionStatus_AllValues_AreValid()
    {
        // Act
        var statuses = (BenchmarkExecutionStatus[])Enum.GetValues(typeof(BenchmarkExecutionStatus));

        // Assert
        Assert.Equal(3, statuses.Length);

        Assert.Contains(BenchmarkExecutionStatus.Succeeded, (IEnumerable<BenchmarkExecutionStatus>)statuses);
        Assert.Contains(BenchmarkExecutionStatus.Failed, (IEnumerable<BenchmarkExecutionStatus>)statuses);
        Assert.Contains(BenchmarkExecutionStatus.Skipped, (IEnumerable<BenchmarkExecutionStatus>)statuses);
    }

    #endregion

    #region BenchmarkSuiteKind Enum Tests

    [Fact]
    public void BenchmarkSuiteKind_AllValues_AreValid()
    {
        // Act
        var kinds = (BenchmarkSuiteKind[])Enum.GetValues(typeof(BenchmarkSuiteKind));

        // Assert
        Assert.Equal(3, kinds.Length);

        Assert.Contains(BenchmarkSuiteKind.Reasoning, (IEnumerable<BenchmarkSuiteKind>)kinds);
        Assert.Contains(BenchmarkSuiteKind.DatasetSuite, (IEnumerable<BenchmarkSuiteKind>)kinds);
        Assert.Contains(BenchmarkSuiteKind.System, (IEnumerable<BenchmarkSuiteKind>)kinds);
    }

    #endregion

    #region Integration Scenarios

    [Fact]
    public void BenchmarkReport_CompleteScenario_AllPropertiesWork()
    {
        // Arrange - Create a complete benchmark report scenario
        var startTime = new DateTimeOffset(2024, 1, 15, 10, 0, 0, TimeSpan.Zero);
        var endTime = new DateTimeOffset(2024, 1, 15, 10, 30, 0, TimeSpan.Zero);

        var mathMetrics = new[]
        {
            new BenchmarkMetricValue { Metric = BenchmarkMetric.Accuracy, Value = 0.85 },
            new BenchmarkMetricValue { Metric = BenchmarkMetric.TotalEvaluated, Value = 100 },
            new BenchmarkMetricValue { Metric = BenchmarkMetric.CorrectCount, Value = 85 },
            new BenchmarkMetricValue { Metric = BenchmarkMetric.AverageConfidence, Value = 0.78 },
            new BenchmarkMetricValue { Metric = BenchmarkMetric.TotalDurationMilliseconds, Value = 600000 },
            new BenchmarkMetricValue { Metric = BenchmarkMetric.AverageTimePerItemMilliseconds, Value = 6000 }
        };

        var mathCategories = new[]
        {
            new BenchmarkCategoryResult { Category = "Algebra", Accuracy = 0.9 },
            new BenchmarkCategoryResult { Category = "Geometry", Accuracy = 0.8 },
            new BenchmarkCategoryResult { Category = "Calculus", Accuracy = 0.75 }
        };

        var gsm8kReport = new BenchmarkSuiteReport();
        SetInternalProperty(gsm8kReport, nameof(BenchmarkSuiteReport.Suite), BenchmarkSuite.GSM8K);
        SetInternalProperty(gsm8kReport, nameof(BenchmarkSuiteReport.Kind), BenchmarkSuiteKind.Reasoning);
        SetInternalProperty(gsm8kReport, nameof(BenchmarkSuiteReport.Name), "GSM8K");
        SetInternalProperty(gsm8kReport, nameof(BenchmarkSuiteReport.Status), BenchmarkExecutionStatus.Succeeded);
        SetInternalProperty(gsm8kReport, nameof(BenchmarkSuiteReport.StartedUtc), startTime);
        SetInternalProperty(gsm8kReport, nameof(BenchmarkSuiteReport.EndedUtc), startTime.AddMinutes(15));
        SetInternalProperty(gsm8kReport, nameof(BenchmarkSuiteReport.Metrics), (IReadOnlyList<BenchmarkMetricValue>)mathMetrics);
        SetInternalProperty(gsm8kReport, nameof(BenchmarkSuiteReport.CategoryAccuracies), (IReadOnlyList<BenchmarkCategoryResult>)mathCategories);

        var mmluReport = new BenchmarkSuiteReport();
        SetInternalProperty(mmluReport, nameof(BenchmarkSuiteReport.Suite), BenchmarkSuite.MMLU);
        SetInternalProperty(mmluReport, nameof(BenchmarkSuiteReport.Kind), BenchmarkSuiteKind.Reasoning);
        SetInternalProperty(mmluReport, nameof(BenchmarkSuiteReport.Name), "MMLU");
        SetInternalProperty(mmluReport, nameof(BenchmarkSuiteReport.Status), BenchmarkExecutionStatus.Succeeded);
        SetInternalProperty(mmluReport, nameof(BenchmarkSuiteReport.StartedUtc), startTime.AddMinutes(15));
        SetInternalProperty(mmluReport, nameof(BenchmarkSuiteReport.EndedUtc), endTime);

        var report = new BenchmarkReport();
        SetInternalProperty(report, nameof(BenchmarkReport.StartedUtc), startTime);
        SetInternalProperty(report, nameof(BenchmarkReport.EndedUtc), endTime);
        SetInternalProperty(report, nameof(BenchmarkReport.Suites), (IReadOnlyList<BenchmarkSuiteReport>)new[] { gsm8kReport, mmluReport });

        // Assert
        Assert.Equal(TimeSpan.FromMinutes(30), report.TotalDuration);
        Assert.Equal(BenchmarkExecutionStatus.Succeeded, report.OverallStatus);
        Assert.Equal(2, report.Suites.Count);

        var gsm8k = report.Suites[0];
        Assert.Equal(BenchmarkSuite.GSM8K, gsm8k.Suite);
        Assert.Equal(TimeSpan.FromMinutes(15), gsm8k.Duration);
        Assert.Equal(6, gsm8k.Metrics.Count);
        Assert.NotNull(gsm8k.CategoryAccuracies);
        Assert.Equal(3, gsm8k.CategoryAccuracies.Count);

        var accuracyMetric = gsm8k.Metrics.First(m => m.Metric == BenchmarkMetric.Accuracy);
        Assert.Equal(0.85, accuracyMetric.Value, precision: 2);
    }

    [Fact]
    public void BenchmarkReport_FailureScenario_HandlesFailuresCorrectly()
    {
        // Arrange
        var startTime = DateTimeOffset.UtcNow;

        var failedReport = new BenchmarkSuiteReport();
        SetInternalProperty(failedReport, nameof(BenchmarkSuiteReport.Suite), BenchmarkSuite.GSM8K);
        SetInternalProperty(failedReport, nameof(BenchmarkSuiteReport.Kind), BenchmarkSuiteKind.Reasoning);
        SetInternalProperty(failedReport, nameof(BenchmarkSuiteReport.Name), "GSM8K");
        SetInternalProperty(failedReport, nameof(BenchmarkSuiteReport.Status), BenchmarkExecutionStatus.Failed);
        SetInternalProperty(failedReport, nameof(BenchmarkSuiteReport.FailureReason), "API rate limit exceeded");
        SetInternalProperty(failedReport, nameof(BenchmarkSuiteReport.StartedUtc), startTime);
        SetInternalProperty(failedReport, nameof(BenchmarkSuiteReport.EndedUtc), startTime.AddSeconds(5));
        SetInternalProperty(failedReport, nameof(BenchmarkSuiteReport.Metrics), Array.Empty<BenchmarkMetricValue>());

        var report = new BenchmarkReport();
        SetInternalProperty(report, nameof(BenchmarkReport.StartedUtc), startTime);
        SetInternalProperty(report, nameof(BenchmarkReport.EndedUtc), startTime.AddSeconds(5));
        SetInternalProperty(report, nameof(BenchmarkReport.Suites), (IReadOnlyList<BenchmarkSuiteReport>)new[] { failedReport });

        // Assert
        Assert.Equal(BenchmarkExecutionStatus.Failed, report.OverallStatus);
        Assert.Single(report.Suites);

        var suite = report.Suites[0];
        Assert.Equal(BenchmarkExecutionStatus.Failed, suite.Status);
        Assert.Equal("API rate limit exceeded", suite.FailureReason);
        Assert.Empty(suite.Metrics);
    }

    #endregion

    #region Helper Methods

    private static BenchmarkSuiteReport CreateSuiteReport(BenchmarkSuite suite, BenchmarkExecutionStatus status)
    {
        var report = new BenchmarkSuiteReport();
        SetInternalProperty(report, nameof(BenchmarkSuiteReport.Suite), suite);
        SetInternalProperty(report, nameof(BenchmarkSuiteReport.Kind), BenchmarkSuiteRegistry.GetSuiteKind(suite));
        SetInternalProperty(report, nameof(BenchmarkSuiteReport.Name), BenchmarkSuiteRegistry.GetDisplayName(suite));
        SetInternalProperty(report, nameof(BenchmarkSuiteReport.Status), status);
        return report;
    }

    private static void SetInternalProperty<TObject, TValue>(TObject obj, string propertyName, TValue value)
    {
        var property = typeof(TObject).GetProperty(propertyName);
        if (property is null)
        {
            throw new InvalidOperationException($"Property {propertyName} not found on type {typeof(TObject).Name}");
        }

        property.SetValue(obj, value);
    }

    #endregion
}
