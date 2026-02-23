using AiDotNet.Evaluation.Statistics;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Engines;

/// <summary>
/// Engine for performing statistical tests on model comparison results.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When comparing machine learning models, you need statistical tests
/// to determine if one model is truly better or if the difference is just due to random chance.
/// This engine provides methods for:
/// <list type="bullet">
/// <item>Comparing two models (paired t-test, Wilcoxon, McNemar)</item>
/// <item>Comparing multiple models (Friedman test)</item>
/// <item>Post-hoc analysis (Nemenyi test)</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class StatisticalTestEngine<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly PairedTTest<T> _pairedTTest;
    private readonly WilcoxonSignedRankTest<T> _wilcoxonTest;
    private readonly McNemarTest<T> _mcnemarTest;
    private readonly FriedmanTest<T> _friedmanTest;
    private readonly NemenyiPostHocTest<T> _nemenyiTest;

    /// <summary>
    /// Initializes the statistical test engine with all available tests.
    /// </summary>
    public StatisticalTestEngine()
    {
        _pairedTTest = new PairedTTest<T>();
        _wilcoxonTest = new WilcoxonSignedRankTest<T>();
        _mcnemarTest = new McNemarTest<T>();
        _friedmanTest = new FriedmanTest<T>();
        _nemenyiTest = new NemenyiPostHocTest<T>();
    }

    /// <summary>
    /// Compares two models using paired t-test (parametric).
    /// </summary>
    /// <remarks>
    /// <para><b>When to use:</b> When you have paired performance scores (e.g., from cross-validation)
    /// and the differences are approximately normally distributed.</para>
    /// </remarks>
    /// <param name="scoresModel1">Performance scores from model 1 (e.g., accuracy per fold).</param>
    /// <param name="scoresModel2">Performance scores from model 2 (same folds).</param>
    /// <param name="alpha">Significance level. Default is 0.05.</param>
    /// <returns>Test result indicating if there's a significant difference.</returns>
    public StatisticalTestResult<T> CompareTwoModelsPairedTTest(T[] scoresModel1, T[] scoresModel2, double alpha = 0.05)
    {
        return _pairedTTest.Test(scoresModel1, scoresModel2, alpha);
    }

    /// <summary>
    /// Compares two models using Wilcoxon signed-rank test (non-parametric).
    /// </summary>
    /// <remarks>
    /// <para><b>When to use:</b> When you have paired scores but cannot assume normality of differences.
    /// This is the safer choice when you're unsure about distributional assumptions.</para>
    /// </remarks>
    public StatisticalTestResult<T> CompareTwoModelsWilcoxon(T[] scoresModel1, T[] scoresModel2, double alpha = 0.05)
    {
        return _wilcoxonTest.Test(scoresModel1, scoresModel2, alpha);
    }

    /// <summary>
    /// Compares two classifiers using McNemar's test on their predictions.
    /// </summary>
    /// <remarks>
    /// <para><b>When to use:</b> When you have binary correct/incorrect predictions from two classifiers
    /// on the same test set. More powerful than comparing accuracy scores alone.</para>
    /// </remarks>
    /// <param name="correctModel1">Boolean array where true = Model 1 predicted correctly.</param>
    /// <param name="correctModel2">Boolean array where true = Model 2 predicted correctly.</param>
    /// <param name="alpha">Significance level.</param>
    public StatisticalTestResult<T> CompareTwoClassifiersMcNemar(T[] correctModel1, T[] correctModel2, double alpha = 0.05)
    {
        return _mcnemarTest.Test(correctModel1, correctModel2, alpha);
    }

    /// <summary>
    /// Compares multiple models using the Friedman test.
    /// </summary>
    /// <remarks>
    /// <para><b>When to use:</b> When comparing 3+ models across multiple datasets or cross-validation folds.
    /// This is the recommended test for ML algorithm comparison (Demsar 2006).</para>
    /// </remarks>
    /// <param name="scores">Performance scores: scores[modelIndex][datasetIndex].</param>
    /// <param name="alpha">Significance level.</param>
    public StatisticalTestResult<T> CompareMultipleModelsFriedman(T[][] scores, double alpha = 0.05)
    {
        return _friedmanTest.Test(scores, alpha);
    }

    /// <summary>
    /// Performs Nemenyi post-hoc test after significant Friedman test.
    /// </summary>
    /// <remarks>
    /// <para><b>When to use:</b> After Friedman test shows significant differences, use Nemenyi
    /// to identify which specific pairs of models differ significantly.</para>
    /// </remarks>
    public NemenyiResult<T> PostHocNemenyi(T[][] scores, double alpha = 0.05)
    {
        return _nemenyiTest.Test(scores, alpha);
    }

    /// <summary>
    /// Performs full comparison of multiple models with automatic test selection.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the recommended workflow for comparing models:
    /// <list type="number">
    /// <item>Friedman test to check if any significant differences exist</item>
    /// <item>If significant, Nemenyi post-hoc to find which pairs differ</item>
    /// </list>
    /// </para>
    /// </remarks>
    /// <param name="scores">Performance scores: scores[modelIndex][datasetIndex].</param>
    /// <param name="modelNames">Optional names for the models.</param>
    /// <param name="alpha">Significance level.</param>
    public ModelComparisonReport<T> CompareModels(T[][] scores, string[]? modelNames = null, double alpha = 0.05)
    {
        int numModels = scores.Length;
        var names = modelNames ?? Enumerable.Range(1, numModels).Select(i => $"Model {i}").ToArray();

        if (numModels < 2)
            throw new ArgumentException("Need at least 2 models to compare.");

        // Two models: use paired tests
        if (numModels == 2)
        {
            var tTestResult = _pairedTTest.Test(scores[0], scores[1], alpha);
            var wilcoxonResult = _wilcoxonTest.Test(scores[0], scores[1], alpha);

            return new ModelComparisonReport<T>
            {
                ModelNames = names,
                NumModels = numModels,
                NumDatasets = scores[0].Length,
                PrimaryTest = wilcoxonResult, // Non-parametric as default
                AdditionalTests = new Dictionary<string, StatisticalTestResult<T>>
                {
                    ["Paired t-test"] = tTestResult,
                    ["Wilcoxon"] = wilcoxonResult
                },
                Summary = GenerateTwoModelSummary(names, tTestResult, wilcoxonResult)
            };
        }

        // Multiple models: Friedman + Nemenyi
        var friedmanResult = _friedmanTest.Test(scores, alpha);
        NemenyiResult<T>? nemenyiResult = null;

        if (friedmanResult.IsSignificant)
        {
            nemenyiResult = _nemenyiTest.Test(scores, alpha);
        }

        return new ModelComparisonReport<T>
        {
            ModelNames = names,
            NumModels = numModels,
            NumDatasets = scores[0].Length,
            PrimaryTest = friedmanResult,
            NemenyiResult = nemenyiResult,
            Summary = GenerateMultiModelSummary(names, friedmanResult, nemenyiResult)
        };
    }

    private string GenerateTwoModelSummary(string[] names, StatisticalTestResult<T> tTest, StatisticalTestResult<T> wilcoxon)
    {
        var lines = new List<string>
        {
            $"Comparison of {names[0]} vs {names[1]}:",
            $"  Paired t-test: p = {tTest.PValue:F4} ({(tTest.IsSignificant ? "significant" : "not significant")})",
            $"  Wilcoxon test: p = {wilcoxon.PValue:F4} ({(wilcoxon.IsSignificant ? "significant" : "not significant")})"
        };

        if (tTest.IsSignificant || wilcoxon.IsSignificant)
        {
            lines.Add($"  Conclusion: There IS a statistically significant difference between the models.");
        }
        else
        {
            lines.Add($"  Conclusion: No statistically significant difference detected.");
        }

        return string.Join(Environment.NewLine, lines);
    }

    private string GenerateMultiModelSummary(string[] names, StatisticalTestResult<T> friedman, NemenyiResult<T>? nemenyi)
    {
        var lines = new List<string>
        {
            $"Comparison of {names.Length} models:",
            $"  Friedman test: p = {friedman.PValue:F4} ({(friedman.IsSignificant ? "significant" : "not significant")})"
        };

        if (!friedman.IsSignificant)
        {
            lines.Add("  Conclusion: No statistically significant differences among models.");
        }
        else if (nemenyi != null)
        {
            lines.Add("  Post-hoc Nemenyi test results:");
            lines.Add($"    Critical Difference: {nemenyi.CriticalDifference:F4}");

            var sigPairs = nemenyi.GetSignificantPairs().ToList();
            if (sigPairs.Count > 0)
            {
                lines.Add("    Significant differences found between:");
                foreach (var (a, b) in sigPairs)
                {
                    lines.Add($"      - {names[a]} vs {names[b]}");
                }
            }
        }

        return string.Join(Environment.NewLine, lines);
    }
}

/// <summary>
/// Comprehensive report from model comparison.
/// </summary>
public class ModelComparisonReport<T>
{
    /// <summary>
    /// Names of the compared models.
    /// </summary>
    public string[] ModelNames { get; init; } = Array.Empty<string>();

    /// <summary>
    /// Number of models compared.
    /// </summary>
    public int NumModels { get; init; }

    /// <summary>
    /// Number of datasets/folds used for comparison.
    /// </summary>
    public int NumDatasets { get; init; }

    /// <summary>
    /// Primary statistical test result.
    /// </summary>
    public StatisticalTestResult<T> PrimaryTest { get; init; } = default!;

    /// <summary>
    /// Additional test results, if computed.
    /// </summary>
    public Dictionary<string, StatisticalTestResult<T>>? AdditionalTests { get; init; }

    /// <summary>
    /// Nemenyi post-hoc results (for multiple model comparison).
    /// </summary>
    public NemenyiResult<T>? NemenyiResult { get; init; }

    /// <summary>
    /// Human-readable summary of the comparison.
    /// </summary>
    public string Summary { get; init; } = "";
}
