namespace AiDotNet.Preprocessing.FeatureSelection;

/// <summary>
/// Helper class for statistical tests used in feature selection.
/// </summary>
internal static class StatisticalTestHelper
{
    /// <summary>
    /// Computes the score and p-value for a feature.
    /// </summary>
    public static (double Score, double PValue) ComputeScore(double[] x, double[] y, SelectKBestScoreFunc scoringFunction)
    {
        int n = x.Length;

        switch (scoringFunction)
        {
            case SelectKBestScoreFunc.FRegression:
                return ComputeFRegression(x, y, n);

            case SelectKBestScoreFunc.MutualInfoRegression:
                return (ComputeMutualInfo(x, y, n), 0);

            case SelectKBestScoreFunc.FClassif:
                return ComputeFClassif(x, y, n);

            case SelectKBestScoreFunc.Chi2:
                return ComputeChi2(x, y, n);

            default:
                return ComputeFRegression(x, y, n);
        }
    }

    private static (double Score, double PValue) ComputeFRegression(double[] x, double[] y, int n)
    {
        double xMean = x.Average();
        double yMean = y.Average();

        double ssXY = 0, ssXX = 0, ssYY = 0;
        for (int i = 0; i < n; i++)
        {
            double dx = x[i] - xMean;
            double dy = y[i] - yMean;
            ssXY += dx * dy;
            ssXX += dx * dx;
            ssYY += dy * dy;
        }

        if (ssXX < 1e-10 || ssYY < 1e-10)
        {
            return (0, 1);
        }

        double r = ssXY / Math.Sqrt(ssXX * ssYY);
        double r2 = r * r;

        int df1 = 1;
        int df2 = n - 2;

        if (df2 <= 0)
        {
            return (0, 1);
        }

        double fStat = (r2 / df1) / ((1 - r2) / df2);
        double pValue = 1.0 - FDistributionCdf(fStat, df1, df2);

        return (fStat, pValue);
    }

    private static (double Score, double PValue) ComputeFClassif(double[] x, double[] y, int n)
    {
        var classes = y.Distinct().OrderBy(c => c).ToArray();
        int nClasses = classes.Length;

        if (nClasses < 2)
        {
            return (0, 1);
        }

        double grandMean = x.Average();

        double ssBetween = 0;
        double ssWithin = 0;

        foreach (double c in classes)
        {
            var classX = new List<double>();
            for (int i = 0; i < n; i++)
            {
                if (Math.Abs(y[i] - c) < 1e-10)
                {
                    classX.Add(x[i]);
                }
            }

            if (classX.Count == 0) continue;

            double classMean = classX.Average();
            ssBetween += classX.Count * (classMean - grandMean) * (classMean - grandMean);

            foreach (double v in classX)
            {
                ssWithin += (v - classMean) * (v - classMean);
            }
        }

        int df1 = nClasses - 1;
        int df2 = n - nClasses;

        if (df1 <= 0 || df2 <= 0 || ssWithin < 1e-10)
        {
            return (0, 1);
        }

        double fStat = (ssBetween / df1) / (ssWithin / df2);
        double pValue = 1.0 - FDistributionCdf(fStat, df1, df2);

        return (fStat, pValue);
    }

    private static (double Score, double PValue) ComputeChi2(double[] x, double[] y, int n)
    {
        var classes = y.Distinct().OrderBy(c => c).ToArray();
        int nClasses = classes.Length;

        if (nClasses < 2)
        {
            return (0, 1);
        }

        double xMin = x.Min();
        if (xMin < 0)
        {
            return (0, 1);
        }

        double[] sumPerClass = new double[nClasses];
        int[] countPerClass = new int[nClasses];
        double totalSum = 0;

        for (int i = 0; i < n; i++)
        {
            int classIdx = Array.IndexOf(classes, y[i]);
            sumPerClass[classIdx] += x[i];
            countPerClass[classIdx]++;
            totalSum += x[i];
        }

        if (totalSum < 1e-10)
        {
            return (0, 1);
        }

        double chi2 = 0;
        for (int c = 0; c < nClasses; c++)
        {
            double expected = totalSum * countPerClass[c] / n;
            if (expected > 1e-10)
            {
                double diff = sumPerClass[c] - expected;
                chi2 += diff * diff / expected;
            }
        }

        int df = nClasses - 1;
        double pValue = 1.0 - ChiSquaredCdf(chi2, df);

        return (chi2, pValue);
    }

    private static double ComputeMutualInfo(double[] x, double[] y, int n)
    {
        int nBins = (int)Math.Sqrt(n);
        nBins = Math.Max(2, Math.Min(20, nBins));

        double xMin = x.Min();
        double xMax = x.Max();
        double yMin = y.Min();
        double yMax = y.Max();

        if (xMax - xMin < 1e-10 || yMax - yMin < 1e-10)
        {
            return 0;
        }

        var joint = new int[nBins, nBins];
        var marginalX = new int[nBins];
        var marginalY = new int[nBins];

        for (int i = 0; i < n; i++)
        {
            int xBin = (int)((x[i] - xMin) / (xMax - xMin) * (nBins - 1));
            int yBin = (int)((y[i] - yMin) / (yMax - yMin) * (nBins - 1));
            xBin = Math.Max(0, Math.Min(nBins - 1, xBin));
            yBin = Math.Max(0, Math.Min(nBins - 1, yBin));

            joint[xBin, yBin]++;
            marginalX[xBin]++;
            marginalY[yBin]++;
        }

        double mi = 0;
        for (int i = 0; i < nBins; i++)
        {
            for (int j = 0; j < nBins; j++)
            {
                if (joint[i, j] > 0 && marginalX[i] > 0 && marginalY[j] > 0)
                {
                    double pxy = (double)joint[i, j] / n;
                    double px = (double)marginalX[i] / n;
                    double py = (double)marginalY[j] / n;
                    mi += pxy * Math.Log(pxy / (px * py));
                }
            }
        }

        return Math.Max(0, mi);
    }

    public static double FDistributionCdf(double x, int df1, int df2)
    {
        if (x <= 0) return 0;
        double a = df1 / 2.0;
        double b = df2 / 2.0;
        double z = df1 * x / (df1 * x + df2);
        return IncompleteBeta(a, b, z);
    }

    public static double ChiSquaredCdf(double x, int df)
    {
        if (x <= 0) return 0;
        return IncompleteGamma(df / 2.0, x / 2.0);
    }

    private static double IncompleteBeta(double a, double b, double x)
    {
        if (x <= 0) return 0;
        if (x >= 1) return 1;

        double sum = 0;
        double term = 1;
        int maxIter = 100;

        for (int n = 0; n < maxIter; n++)
        {
            sum += term;
            term *= (a + n) / (a + b + n) * x * (n + 1) / (n + 1);
            if (Math.Abs(term) < 1e-10) break;
        }

        return Math.Pow(x, a) * Math.Pow(1 - x, b) * sum / (a * Beta(a, b));
    }

    private static double Beta(double a, double b)
    {
        return Math.Exp(LogGamma(a) + LogGamma(b) - LogGamma(a + b));
    }

    private static double IncompleteGamma(double a, double x)
    {
        if (x <= 0) return 0;

        double sum = 0;
        double term = 1.0 / a;
        int maxIter = 100;

        for (int n = 0; n < maxIter; n++)
        {
            sum += term;
            term *= x / (a + n + 1);
            if (Math.Abs(term) < 1e-10) break;
        }

        return Math.Exp(-x + a * Math.Log(x) - LogGamma(a)) * sum;
    }

    private static double LogGamma(double x)
    {
        double[] c = { 76.18009172947146, -86.50532032941677, 24.01409824083091,
                       -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5 };

        double y = x;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.Log(tmp);

        double ser = 1.000000000190015;
        for (int j = 0; j < 6; j++)
        {
            y += 1;
            ser += c[j] / y;
        }

        return -tmp + Math.Log(2.5066282746310005 * ser / x);
    }

    /// <summary>
    /// Applies Benjamini-Hochberg correction for multiple testing.
    /// </summary>
    /// <param name="pValues">Original p-values.</param>
    /// <returns>Adjusted p-values (FDR-corrected).</returns>
    public static double[] BenjaminiHochbergCorrection(double[] pValues)
    {
        int n = pValues.Length;
        var indexed = pValues.Select((p, i) => (p, i)).OrderBy(x => x.p).ToArray();
        var adjusted = new double[n];

        double cumMin = 1.0;
        for (int rank = n; rank >= 1; rank--)
        {
            var (p, origIdx) = indexed[rank - 1];
            double adjustedP = p * n / rank;
            cumMin = Math.Min(cumMin, adjustedP);
            adjusted[origIdx] = Math.Min(1.0, cumMin);
        }

        return adjusted;
    }

    /// <summary>
    /// Applies Bonferroni correction for multiple testing.
    /// </summary>
    /// <param name="pValues">Original p-values.</param>
    /// <returns>Adjusted p-values (Bonferroni-corrected).</returns>
    public static double[] BonferroniCorrection(double[] pValues)
    {
        int n = pValues.Length;
        return pValues.Select(p => Math.Min(1.0, p * n)).ToArray();
    }
}
