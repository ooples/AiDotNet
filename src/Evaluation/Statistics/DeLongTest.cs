using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.Statistics;

/// <summary>
/// DeLong's test for comparing two ROC curves.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> DeLong's test compares the AUC of two classifiers:
/// <list type="bullet">
/// <item>Tests if the difference in AUC is statistically significant</item>
/// <item>Specifically designed for comparing ROC curves</item>
/// <item>Non-parametric and doesn't assume normal distribution</item>
/// <item>Can handle correlated data (same test set for both models)</item>
/// </list>
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Comparing two binary classifiers on the same dataset</item>
/// <item>When you want to know if AUC improvement is significant</item>
/// <item>Medical diagnostic test comparisons</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class DeLongTest<T> : IStatisticalTest<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public string Name => "DeLong Test";
    public string Description => "Compares two ROC curves to test if AUC difference is significant.";
    public bool IsPaired => true;
    public bool IsNonParametric => true;

    /// <summary>
    /// Performs DeLong's test comparing two sets of predicted probabilities.
    /// </summary>
    /// <param name="predictions1">Probabilities from first classifier.</param>
    /// <param name="predictions2">Probabilities from second classifier.</param>
    /// <param name="actuals">True binary labels.</param>
    /// <returns>Statistical test result with p-value.</returns>
    public StatisticalTestResult<T> Test(T[] predictions1, T[] predictions2, T[] actuals)
    {
        if (predictions1.Length != predictions2.Length || predictions1.Length != actuals.Length)
            throw new ArgumentException("All arrays must have the same length.");

        int n = predictions1.Length;

        // Separate positives and negatives
        var posIndices = new List<int>();
        var negIndices = new List<int>();

        for (int i = 0; i < n; i++)
        {
            if (NumOps.ToDouble(actuals[i]) >= 0.5)
                posIndices.Add(i);
            else
                negIndices.Add(i);
        }

        int nPos = posIndices.Count;
        int nNeg = negIndices.Count;

        if (nPos == 0 || nNeg == 0)
        {
            return new StatisticalTestResult<T>
            {
                TestName = Name,
                Statistic = NumOps.Zero,
                PValue = NumOps.One,
                IsSignificant = false,
                EffectSize = NumOps.Zero,
                Description = "Cannot compute: need both positive and negative samples."
            };
        }

        // Compute AUC for both models using Mann-Whitney U statistic
        double auc1 = ComputeAUC(predictions1, actuals, posIndices, negIndices);
        double auc2 = ComputeAUC(predictions2, actuals, posIndices, negIndices);

        // Compute structural components (V matrices)
        double[] V10_1 = ComputeV10(predictions1, posIndices, negIndices);
        double[] V10_2 = ComputeV10(predictions2, posIndices, negIndices);
        double[] V01_1 = ComputeV01(predictions1, posIndices, negIndices);
        double[] V01_2 = ComputeV01(predictions2, posIndices, negIndices);

        // Compute variance-covariance matrix
        double s10_11 = Covariance(V10_1, V10_1);
        double s10_22 = Covariance(V10_2, V10_2);
        double s10_12 = Covariance(V10_1, V10_2);

        double s01_11 = Covariance(V01_1, V01_1);
        double s01_22 = Covariance(V01_2, V01_2);
        double s01_12 = Covariance(V01_1, V01_2);

        double var1 = s10_11 / nPos + s01_11 / nNeg;
        double var2 = s10_22 / nPos + s01_22 / nNeg;
        double covar = s10_12 / nPos + s01_12 / nNeg;

        double varDiff = var1 + var2 - 2 * covar;
        if (varDiff < 1e-10)
        {
            return new StatisticalTestResult<T>
            {
                TestName = Name,
                Statistic = NumOps.Zero,
                PValue = NumOps.One,
                IsSignificant = false,
                EffectSize = NumOps.FromDouble(auc1 - auc2),
                Description = "Variance is zero; cannot compute test statistic."
            };
        }

        double z = (auc1 - auc2) / Math.Sqrt(varDiff);
        double pValue = 2 * (1 - NormalCDF(Math.Abs(z)));

        return new StatisticalTestResult<T>
        {
            TestName = Name,
            Statistic = NumOps.FromDouble(z),
            PValue = NumOps.FromDouble(pValue),
            IsSignificant = pValue < 0.05,
            EffectSize = NumOps.FromDouble(auc1 - auc2),
            Interpretation = pValue < 0.05
                ? $"Significant difference in AUC (AUC1={auc1:F4}, AUC2={auc2:F4}, Δ={auc1 - auc2:F4})"
                : $"No significant difference in AUC (AUC1={auc1:F4}, AUC2={auc2:F4})",
            Description = "DeLong's test for comparing AUC of two ROC curves."
        };
    }

    private double ComputeAUC(T[] predictions, T[] actuals, List<int> posIndices, List<int> negIndices)
    {
        double sum = 0;
        foreach (int posIdx in posIndices)
        {
            double posProb = NumOps.ToDouble(predictions[posIdx]);
            foreach (int negIdx in negIndices)
            {
                double negProb = NumOps.ToDouble(predictions[negIdx]);
                if (posProb > negProb) sum += 1;
                else if (Math.Abs(posProb - negProb) < 1e-10) sum += 0.5;
            }
        }
        return sum / (posIndices.Count * negIndices.Count);
    }

    private double[] ComputeV10(T[] predictions, List<int> posIndices, List<int> negIndices)
    {
        var result = new double[posIndices.Count];
        for (int i = 0; i < posIndices.Count; i++)
        {
            double posProb = NumOps.ToDouble(predictions[posIndices[i]]);
            double sum = 0;
            foreach (int negIdx in negIndices)
            {
                double negProb = NumOps.ToDouble(predictions[negIdx]);
                if (posProb > negProb) sum += 1;
                else if (Math.Abs(posProb - negProb) < 1e-10) sum += 0.5;
            }
            result[i] = sum / negIndices.Count;
        }
        return result;
    }

    private double[] ComputeV01(T[] predictions, List<int> posIndices, List<int> negIndices)
    {
        var result = new double[negIndices.Count];
        for (int i = 0; i < negIndices.Count; i++)
        {
            double negProb = NumOps.ToDouble(predictions[negIndices[i]]);
            double sum = 0;
            foreach (int posIdx in posIndices)
            {
                double posProb = NumOps.ToDouble(predictions[posIdx]);
                if (posProb > negProb) sum += 1;
                else if (Math.Abs(posProb - negProb) < 1e-10) sum += 0.5;
            }
            result[i] = sum / posIndices.Count;
        }
        return result;
    }

    private double Covariance(double[] x, double[] y)
    {
        if (x.Length != y.Length || x.Length == 0) return 0;
        double meanX = x.Average();
        double meanY = y.Average();
        double sum = 0;
        for (int i = 0; i < x.Length; i++)
        {
            sum += (x[i] - meanX) * (y[i] - meanY);
        }
        return sum / (x.Length - 1);
    }

    private double NormalCDF(double z)
    {
        // Approximation of standard normal CDF
        double t = 1.0 / (1.0 + 0.2316419 * Math.Abs(z));
        double d = 0.3989423 * Math.Exp(-z * z / 2);
        double prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        return z > 0 ? 1 - prob : prob;
    }
}
