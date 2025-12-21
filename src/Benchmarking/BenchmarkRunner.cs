using AiDotNet.Benchmarking.Models;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Results;
using AiDotNet.Reasoning.Benchmarks.Models;

namespace AiDotNet.Benchmarking;

/// <summary>
/// Internal benchmark runner that executes benchmark suites and produces structured reports.
/// </summary>
internal static class BenchmarkRunner
{
    public static async Task<BenchmarkReport> RunAsync<T, TInput, TOutput>(
        PredictionModelResult<T, TInput, TOutput> model,
        BenchmarkingOptions options,
        CancellationToken cancellationToken = default)
    {
        if (model is null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        if (options.Suites is null)
        {
            throw new ArgumentException("BenchmarkingOptions.Suites cannot be null.", nameof(options));
        }

        if (options.Suites.Length == 0)
        {
            return new BenchmarkReport
            {
                StartedUtc = DateTimeOffset.UtcNow,
                EndedUtc = DateTimeOffset.UtcNow,
                Suites = Array.Empty<BenchmarkSuiteReport>()
            };
        }

        var startedUtc = DateTimeOffset.UtcNow;
        var suiteReports = new List<BenchmarkSuiteReport>(options.Suites.Length);
        var failures = new List<Exception>();

        foreach (var suite in options.Suites)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var suiteStartedUtc = DateTimeOffset.UtcNow;
            try
            {
                var suiteReport = await RunSuiteAsync(model, suite, options, cancellationToken).ConfigureAwait(false);
                suiteReport.StartedUtc = suiteStartedUtc;
                suiteReport.EndedUtc = DateTimeOffset.UtcNow;
                suiteReports.Add(suiteReport);
            }
            catch (Exception ex)
            {
                var failedReport = new BenchmarkSuiteReport
                {
                    Suite = suite,
                    Kind = BenchmarkSuiteRegistry.GetSuiteKind(suite),
                    Name = BenchmarkSuiteRegistry.GetDisplayName(suite),
                    Status = BenchmarkExecutionStatus.Failed,
                    FailureReason = ex.Message,
                    StartedUtc = suiteStartedUtc,
                    EndedUtc = DateTimeOffset.UtcNow,
                    Metrics = Array.Empty<BenchmarkMetricValue>()
                };

                suiteReports.Add(failedReport);

                if (options.FailurePolicy == BenchmarkFailurePolicy.FailFast)
                {
                    throw;
                }

                failures.Add(ex);
            }
        }

        var endedUtc = DateTimeOffset.UtcNow;

        var report = new BenchmarkReport
        {
            StartedUtc = startedUtc,
            EndedUtc = endedUtc,
            Suites = suiteReports
        };

        if (failures.Count > 0 && options.FailurePolicy == BenchmarkFailurePolicy.ContinueAndThrowAggregate)
        {
            throw new AggregateException("One or more benchmark suites failed.", failures);
        }

        return report;
    }

    private static async Task<BenchmarkSuiteReport> RunSuiteAsync<T, TInput, TOutput>(
        PredictionModelResult<T, TInput, TOutput> model,
        BenchmarkSuite suite,
        BenchmarkingOptions options,
        CancellationToken cancellationToken)
    {
        var kind = BenchmarkSuiteRegistry.GetSuiteKind(suite);
        if (kind == BenchmarkSuiteKind.DatasetSuite)
        {
            if (suite == BenchmarkSuite.LEAF)
            {
                var leaf = options.Leaf ?? throw new InvalidOperationException(
                    "BenchmarkingOptions.Leaf must be provided when running BenchmarkSuite.LEAF.");

                return await LeafFederatedBenchmarkSuiteRunner.RunAsync(model, suite, options, leaf, cancellationToken).ConfigureAwait(false);
            }

            if (suite == BenchmarkSuite.FEMNIST)
            {
                var leaf = options.Vision?.Femnist ?? throw new InvalidOperationException(
                    "BenchmarkingOptions.Vision.Femnist must be provided when running BenchmarkSuite.FEMNIST.");

                return await LeafFederatedBenchmarkSuiteRunner.RunAsync(model, suite, options, leaf, cancellationToken).ConfigureAwait(false);
            }

            if (suite == BenchmarkSuite.Sent140)
            {
                var sent140 = options.Text?.Sent140 ?? throw new InvalidOperationException(
                    "BenchmarkingOptions.Text.Sent140 must be provided when running BenchmarkSuite.Sent140.");

                return await Sent140FederatedBenchmarkSuiteRunner.RunAsync(model, suite, options, sent140, cancellationToken).ConfigureAwait(false);
            }

            if (suite == BenchmarkSuite.Shakespeare)
            {
                var shakespeare = options.Text?.Shakespeare ?? throw new InvalidOperationException(
                    "BenchmarkingOptions.Text.Shakespeare must be provided when running BenchmarkSuite.Shakespeare.");

                return await ShakespeareFederatedBenchmarkSuiteRunner.RunAsync(model, suite, options, shakespeare, cancellationToken).ConfigureAwait(false);
            }

            if (suite == BenchmarkSuite.Reddit)
            {
                var reddit = options.Text?.Reddit ?? throw new InvalidOperationException(
                    "BenchmarkingOptions.Text.Reddit must be provided when running BenchmarkSuite.Reddit.");

                return await RedditFederatedBenchmarkSuiteRunner.RunAsync(model, suite, options, reddit, cancellationToken).ConfigureAwait(false);
            }

            if (suite == BenchmarkSuite.StackOverflow)
            {
                var stackOverflow = options.Text?.StackOverflow ?? throw new InvalidOperationException(
                    "BenchmarkingOptions.Text.StackOverflow must be provided when running BenchmarkSuite.StackOverflow.");

                return await StackOverflowFederatedBenchmarkSuiteRunner.RunAsync(model, suite, options, stackOverflow, cancellationToken).ConfigureAwait(false);
            }

            if (suite is BenchmarkSuite.CIFAR10 or BenchmarkSuite.CIFAR100)
            {
                return await CifarFederatedBenchmarkSuiteRunner.RunAsync(model, suite, options, cancellationToken).ConfigureAwait(false);
            }

            if (suite == BenchmarkSuite.TabularNonIID)
            {
                return await SyntheticTabularFederatedBenchmarkSuiteRunner.RunAsync(model, suite, options, cancellationToken).ConfigureAwait(false);
            }

            throw new NotSupportedException($"Dataset benchmark suite '{suite}' is not yet supported by the unified runner.");
        }

        if (kind != BenchmarkSuiteKind.Reasoning)
        {
            throw new NotSupportedException($"Benchmark suite '{suite}' is not yet supported by the unified runner.");
        }

        int? sampleSize = options.SampleSize;
        if (options.CiMode && sampleSize is null)
        {
            sampleSize = 25;
        }

        IBenchmark<double> benchmark = BenchmarkSuiteRegistry.CreateReasoningBenchmark(suite);
        BenchmarkResult<double> result = await model.EvaluateBenchmarkAsync(benchmark, sampleSize, cancellationToken)
            .ConfigureAwait(false);

        var metrics = new List<BenchmarkMetricValue>
        {
            new() { Metric = BenchmarkMetric.TotalEvaluated, Value = result.TotalEvaluated },
            new() { Metric = BenchmarkMetric.CorrectCount, Value = result.CorrectCount },
            new() { Metric = BenchmarkMetric.Accuracy, Value = Convert.ToDouble(result.Accuracy) },
            new() { Metric = BenchmarkMetric.AverageConfidence, Value = Convert.ToDouble(result.AverageConfidence) },
            new() { Metric = BenchmarkMetric.TotalDurationMilliseconds, Value = result.TotalDuration.TotalMilliseconds },
            new() { Metric = BenchmarkMetric.AverageTimePerItemMilliseconds, Value = result.AverageTimePerProblem.TotalMilliseconds }
        };

        IReadOnlyList<BenchmarkCategoryResult>? categoryAccuracies = null;
        if (options.DetailLevel == BenchmarkReportDetailLevel.Detailed && result.AccuracyByCategory.Count > 0)
        {
            categoryAccuracies = result.AccuracyByCategory
                .Select(kvp => new BenchmarkCategoryResult
                {
                    Category = kvp.Key,
                    Accuracy = Convert.ToDouble(kvp.Value)
                })
                .OrderByDescending(x => x.Accuracy)
                .ToList();
        }

        return new BenchmarkSuiteReport
        {
            Suite = suite,
            Kind = kind,
            Name = BenchmarkSuiteRegistry.GetDisplayName(suite),
            Status = BenchmarkExecutionStatus.Succeeded,
            FailureReason = null,
            Metrics = metrics,
            CategoryAccuracies = categoryAccuracies
        };
    }
}
