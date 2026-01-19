using AiDotNet.Enums;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration options for running benchmarks through the AiDotNet facade.
/// </summary>
/// <remarks>
/// <para>
/// This options class is designed for use with <c>AiModelBuilder.ConfigureBenchmarking(...)</c> and
/// <c>AiModelResult.EvaluateBenchmarksAsync(...)</c>. Users specify which benchmark suites to run,
/// and AiDotNet orchestrates the execution behind the scenes.
/// </para>
/// <para>
/// <b>For Beginners:</b> Benchmarking is how you measure model quality using standardized tests.
/// You choose a suite (like GSM8K for math reasoning), and AiDotNet runs it and returns a report.
/// </para>
/// </remarks>
public sealed class BenchmarkingOptions
{
    /// <summary>
    /// Gets or sets the benchmark suites to run.
    /// </summary>
    /// <remarks>
    /// If empty, no benchmarks will be run.
    /// </remarks>
    public BenchmarkSuite[] Suites { get; set; } = Array.Empty<BenchmarkSuite>();

    /// <summary>
    /// Gets or sets an optional sample size for suites that support sampling.
    /// </summary>
    /// <remarks>
    /// If null, suites use their default behavior (often "all problems" or suite-defined sampling).
    /// </remarks>
    public int? SampleSize { get; set; }

    /// <summary>
    /// Gets or sets an optional deterministic seed used for CI-friendly sampling.
    /// </summary>
    /// <remarks>
    /// When specified, suites that support subsampling use this value to ensure repeatable selection.
    /// </remarks>
    public int? Seed { get; set; }

    /// <summary>
    /// Gets or sets whether benchmarking should run in CI-friendly mode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// CI-friendly mode favors smaller sample sizes to keep runs fast and stable.
    /// Deterministic seeding is introduced in Sprint 2 (dataset suites + CI mode).
    /// </para>
    /// </remarks>
    public bool CiMode { get; set; }

    /// <summary>
    /// Gets or sets how much detail should be included in reports.
    /// </summary>
    public BenchmarkReportDetailLevel DetailLevel { get; set; } = BenchmarkReportDetailLevel.Summary;

    /// <summary>
    /// Gets or sets how failures should be handled.
    /// </summary>
    public BenchmarkFailurePolicy FailurePolicy { get; set; } = BenchmarkFailurePolicy.FailFast;

    /// <summary>
    /// Gets or sets whether the generated report should be attached to the model result.
    /// </summary>
    /// <remarks>
    /// When true, the last produced report is stored on the <c>AiModelResult</c> instance and can be retrieved
    /// via <c>AiModelResult.GetBenchmarkReport()</c>.
    /// </remarks>
    public bool AttachReportToResult { get; set; } = true;

    /// <summary>
    /// Gets or sets LEAF federated benchmark configuration (required when running <see cref="BenchmarkSuite.LEAF"/>).
    /// </summary>
    public LeafFederatedBenchmarkOptions? Leaf { get; set; }

    /// <summary>
    /// Gets or sets federated vision benchmark configuration (FEMNIST/CIFAR suites).
    /// </summary>
    public FederatedVisionBenchmarkOptions? Vision { get; set; }

    /// <summary>
    /// Gets or sets federated tabular benchmark configuration (synthetic non-IID suite).
    /// </summary>
    public FederatedTabularBenchmarkOptions? Tabular { get; set; }

    /// <summary>
    /// Gets or sets federated text benchmark configuration (Sent140/Shakespeare suites).
    /// </summary>
    public FederatedTextBenchmarkOptions? Text { get; set; }
}
