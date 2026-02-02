using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Statistics;

/// <summary>
/// McNemar's test: compares the performance of two binary classifiers on the same dataset.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> McNemar's test is specifically designed for comparing two
/// classifiers' predictions on the same dataset. It uses a 2x2 contingency table:
/// <code>
///                  Model B
///              Correct  Wrong
/// Model A  Correct   a      b
///          Wrong     c      d
/// </code>
/// The test focuses on the disagreement cells (b and c) - cases where one model is right
/// and the other is wrong.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Comparing two classifiers on the same test set</item>
/// <item>Binary classification problems</item>
/// <item>When you want to test if classifier A is significantly better than B</item>
/// </list>
/// </para>
/// </remarks>
public class McNemarTest<T> : ITwoSampleTest<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "McNemar's Test";
    public string Description => "Compares performance of two classifiers on the same data.";

    /// <summary>
    /// Performs McNemar's test using binary predictions.
    /// </summary>
    /// <param name="correctA">Boolean array where true = Model A predicted correctly.</param>
    /// <param name="correctB">Boolean array where true = Model B predicted correctly.</param>
    /// <param name="alpha">Significance level.</param>
    /// <returns>Test result.</returns>
    public StatisticalTestResult<T> Test(T[] correctA, T[] correctB, double alpha = 0.05)
    {
        if (correctA.Length != correctB.Length)
            throw new ArgumentException("Samples must have the same length.");
        if (correctA.Length < 10)
            throw new ArgumentException("Need at least 10 observations for McNemar's test.");

        int n = correctA.Length;

        // Build contingency table
        // b = A correct, B wrong
        // c = A wrong, B correct
        int b = 0, c = 0;
        for (int i = 0; i < n; i++)
        {
            bool aCorrect = NumOps.ToDouble(correctA[i]) >= 0.5;
            bool bCorrect = NumOps.ToDouble(correctB[i]) >= 0.5;

            if (aCorrect && !bCorrect) b++;
            else if (!aCorrect && bCorrect) c++;
        }

        // If both b and c are zero, no disagreement exists
        if (b + c == 0)
        {
            return new StatisticalTestResult<T>
            {
                TestName = Name,
                Statistic = NumOps.Zero,
                PValue = NumOps.One,
                IsSignificant = false,
                Alpha = alpha,
                Description = "No disagreement between classifiers."
            };
        }

        // McNemar's chi-squared statistic (with continuity correction)
        double chiSquared = Math.Pow(Math.Abs(b - c) - 1, 2) / (b + c);

        // For small samples (b + c < 25), use exact binomial test
        double pValue;
        if (b + c < 25)
        {
            // Exact test: probability of observing k or more extreme under null
            pValue = ExactMcNemarPValue(b, c);
        }
        else
        {
            // Chi-squared with 1 df
            pValue = ChiSquaredPValue(chiSquared, 1);
        }

        // Odds ratio as effect size
        double oddsRatio = (c > 0) ? (double)b / c : double.PositiveInfinity;

        return new StatisticalTestResult<T>
        {
            TestName = Name,
            Statistic = NumOps.FromDouble(chiSquared),
            PValue = NumOps.FromDouble(pValue),
            IsSignificant = pValue < alpha,
            Alpha = alpha,
            DegreesOfFreedom = 1,
            EffectSize = NumOps.FromDouble(oddsRatio),
            Description = $"b (A>B) = {b}, c (B>A) = {c}, Odds ratio = {oddsRatio:F2}"
        };
    }

    private static double ExactMcNemarPValue(int b, int c)
    {
        int n = b + c;
        int k = Math.Min(b, c);

        // Two-tailed binomial test
        double pValue = 0;
        for (int i = 0; i <= k; i++)
        {
            pValue += BinomialProbability(n, i, 0.5);
        }
        return Math.Min(1.0, 2 * pValue); // Two-tailed, capped at 1.0
    }

    private static double BinomialProbability(int n, int k, double p)
    {
        if (k < 0 || k > n) return 0;
        double logProb = LogBinomialCoefficient(n, k) + k * Math.Log(p) + (n - k) * Math.Log(1 - p);
        return Math.Exp(logProb);
    }

    private static double LogBinomialCoefficient(int n, int k)
    {
        if (k < 0 || k > n) return double.NegativeInfinity;
        return LogFactorial(n) - LogFactorial(k) - LogFactorial(n - k);
    }

    private static double LogFactorial(int n)
    {
        if (n <= 1) return 0;
        double result = 0;
        for (int i = 2; i <= n; i++)
            result += Math.Log(i);
        return result;
    }

    private static double ChiSquaredPValue(double chiSquared, int df)
    {
        // Use incomplete gamma function for chi-squared CDF
        return 1 - IncompleteGamma(df / 2.0, chiSquared / 2.0);
    }

    private static double IncompleteGamma(double a, double x)
    {
        // Regularized incomplete gamma function
        if (x < 0 || a <= 0) return 0;
        if (x == 0) return 0;

        if (x < a + 1)
        {
            // Use series representation
            double sum = 1.0 / a;
            double term = sum;
            for (int n = 1; n < 100; n++)
            {
                term *= x / (a + n);
                sum += term;
                if (Math.Abs(term) < 1e-10 * Math.Abs(sum)) break;
            }
            return sum * Math.Exp(-x + a * Math.Log(x) - LogGamma(a));
        }
        else
        {
            // Use continued fraction
            return 1 - IncompleteGammaComplement(a, x);
        }
    }

    private static double IncompleteGammaComplement(double a, double x)
    {
        double b = x + 1 - a;
        double c = 1e30;
        double d = 1 / b;
        double h = d;

        for (int i = 1; i < 100; i++)
        {
            double an = -i * (i - a);
            b += 2;
            d = an * d + b;
            if (Math.Abs(d) < 1e-30) d = 1e-30;
            c = b + an / c;
            if (Math.Abs(c) < 1e-30) c = 1e-30;
            d = 1 / d;
            double del = d * c;
            h *= del;
            if (Math.Abs(del - 1) < 1e-10) break;
        }

        return h * Math.Exp(-x + a * Math.Log(x) - LogGamma(a));
    }

    private static double LogGamma(double x)
    {
        double[] coef = { 76.18009173, -86.50532033, 24.01409822, -1.231739516, 0.00120858003, -0.00000536382 };
        double y = x;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.Log(tmp);
        double ser = 1.000000000190015;
        for (int j = 0; j < 6; j++)
            ser += coef[j] / ++y;
        return -tmp + Math.Log(2.5066282746310005 * ser / x);
    }
}
