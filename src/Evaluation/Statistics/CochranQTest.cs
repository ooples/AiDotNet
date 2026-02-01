using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Statistics;

/// <summary>
/// Cochran's Q test for comparing multiple classifiers on the same dataset.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Cochran's Q test is an extension of McNemar's test:
/// <list type="bullet">
/// <item>Compares 3+ classifiers on the same binary classification task</item>
/// <item>Tests if all classifiers have the same error rate</item>
/// <item>Non-parametric test for matched samples</item>
/// <item>Uses chi-square distribution for p-value</item>
/// </list>
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Comparing multiple classifiers on the same test set</item>
/// <item>Binary classification only (use Friedman for continuous scores)</item>
/// <item>When samples are matched (same data across all classifiers)</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class CochranQTest<T> : IClassifierComparisonTest<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "Cochran's Q Test";
    public string Description => "Tests if multiple classifiers have equal error rates.";
    public bool IsPaired => true;
    public bool IsNonParametric => true;

    /// <summary>
    /// Tests if multiple classifiers have significantly different error rates.
    /// </summary>
    /// <param name="predictions">Array of prediction arrays, one per classifier.</param>
    /// <param name="actuals">True labels.</param>
    /// <returns>Statistical test result with p-value.</returns>
    public StatisticalTestResult<T> Test(T[][] predictions, T[] actuals)
    {
        int k = predictions.Length; // Number of classifiers
        if (k < 2)
            throw new ArgumentException("Need at least 2 classifiers for Cochran's Q test.");

        int n = actuals.Length; // Number of samples
        foreach (var pred in predictions)
        {
            if (pred.Length != n)
                throw new ArgumentException("All prediction arrays must have the same length as actuals.");
        }

        // Convert to binary success/failure matrix
        // X[i,j] = 1 if classifier j is correct on sample i, 0 otherwise
        var X = new int[n, k];
        for (int i = 0; i < n; i++)
        {
            bool actualPositive = NumOps.ToDouble(actuals[i]) >= 0.5;
            for (int j = 0; j < k; j++)
            {
                bool predPositive = NumOps.ToDouble(predictions[j][i]) >= 0.5;
                X[i, j] = (predPositive == actualPositive) ? 1 : 0;
            }
        }

        // Compute row and column totals
        var rowTotals = new int[n];
        var colTotals = new int[k];
        int grandTotal = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++)
            {
                rowTotals[i] += X[i, j];
                colTotals[j] += X[i, j];
                grandTotal += X[i, j];
            }
        }

        // Cochran's Q statistic
        double sumColSquares = colTotals.Sum(c => (double)c * c);
        double sumRowSquares = rowTotals.Sum(r => (double)r * r);
        double sumRowTotals = rowTotals.Sum();

        double numerator = (k - 1) * (k * sumColSquares - grandTotal * grandTotal);
        double denominator = k * sumRowTotals - sumRowSquares;

        if (Math.Abs(denominator) < 1e-10)
        {
            return new StatisticalTestResult<T>
            {
                TestName = Name,
                Statistic = NumOps.Zero,
                PValue = NumOps.One,
                IsSignificant = false,
                DegreesOfFreedom = k - 1,
                Description = "All classifiers have identical performance."
            };
        }

        double Q = numerator / denominator;
        int df = k - 1;
        double pValue = 1 - ChiSquareCDF(Q, df);

        // Compute accuracy for each classifier
        var accuracies = colTotals.Select(c => (double)c / n).ToArray();

        return new StatisticalTestResult<T>
        {
            TestName = Name,
            Statistic = NumOps.FromDouble(Q),
            PValue = NumOps.FromDouble(pValue),
            IsSignificant = pValue < 0.05,
            DegreesOfFreedom = df,
            Interpretation = pValue < 0.05
                ? $"Significant difference among {k} classifiers (accuracies: {string.Join(", ", accuracies.Select(a => $"{a:P1}"))})"
                : $"No significant difference among {k} classifiers",
            Description = "Cochran's Q test for comparing multiple classifiers on binary outcomes."
        };
    }

    private double ChiSquareCDF(double x, int df)
    {
        if (x <= 0) return 0;

        // Use incomplete gamma function approximation
        double a = df / 2.0;
        double x2 = x / 2.0;

        return LowerIncompleteGamma(a, x2) / GammaFunction(a);
    }

    private double LowerIncompleteGamma(double a, double x)
    {
        if (x < 0) return 0;
        if (x == 0) return 0;

        // Series expansion for small x
        if (x < a + 1)
        {
            double sum = 1.0 / a;
            double term = 1.0 / a;
            for (int n = 1; n < 100; n++)
            {
                term *= x / (a + n);
                sum += term;
                if (Math.Abs(term) < 1e-10 * Math.Abs(sum)) break;
            }
            return Math.Exp(-x + a * Math.Log(x)) * sum;
        }
        else
        {
            // Continued fraction for large x
            return GammaFunction(a) - UpperIncompleteGamma(a, x);
        }
    }

    private double UpperIncompleteGamma(double a, double x)
    {
        // Continued fraction approximation
        double b = x + 1 - a;
        double c = 1.0 / 1e-30;
        double d = 1.0 / b;
        double h = d;

        for (int i = 1; i < 100; i++)
        {
            double an = -i * (i - a);
            b += 2.0;
            d = an * d + b;
            if (Math.Abs(d) < 1e-30) d = 1e-30;
            c = b + an / c;
            if (Math.Abs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d;
            double delta = d * c;
            h *= delta;
            if (Math.Abs(delta - 1.0) < 1e-10) break;
        }

        return Math.Exp(-x + a * Math.Log(x)) * h;
    }

    private double GammaFunction(double x)
    {
        // Lanczos approximation
        double[] coefficients = { 76.18009172947146, -86.50532032941677, 24.01409824083091,
            -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5 };

        double y = x;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.Log(tmp);
        double sum = 1.000000000190015;

        for (int j = 0; j < 6; j++)
        {
            y += 1;
            sum += coefficients[j] / y;
        }

        return Math.Exp(-tmp + Math.Log(2.5066282746310005 * sum / x));
    }
}
