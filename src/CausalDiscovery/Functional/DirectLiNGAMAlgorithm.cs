using AiDotNet.Extensions;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Functional;

/// <summary>
/// DirectLiNGAM — direct method for LiNGAM without ICA.
/// </summary>
/// <remarks>
/// <para>
/// DirectLiNGAM avoids the ICA step entirely and instead uses a direct regression-based
/// approach to identify the causal ordering. It iteratively finds the "root" variable
/// (the one with the most independent residuals) and removes its effect.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Find the variable with minimum dependence on others (root cause)</item>
/// <item>Regress out the root's effect from all remaining variables</item>
/// <item>Repeat on the residuals until all variables are ordered</item>
/// <item>Estimate connection strengths via OLS in the causal order</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> DirectLiNGAM finds causal structure step by step. First, it finds
/// the variable that seems to cause others but isn't caused by anything (the "root").
/// Then it removes that variable's influence and repeats. This gives a causal ordering
/// from which the full structure follows naturally.
/// </para>
/// <para>
/// Reference: Shimizu et al. (2011), "DirectLiNGAM: A Direct Method for Learning a
/// Linear Non-Gaussian Structural Equation Model", JMLR.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DirectLiNGAMAlgorithm<T> : FunctionalBase<T>
{
    private double _threshold = 0.1;

    /// <inheritdoc/>
    public override string Name => "DirectLiNGAM";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes DirectLiNGAM with optional configuration.
    /// </summary>
    public DirectLiNGAMAlgorithm(CausalDiscoveryOptions? options = null)
    {
        if (options?.EdgeThreshold.HasValue == true) _threshold = options.EdgeThreshold.Value;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        var standardized = StandardizeData(data);

        var remaining = Enumerable.Range(0, d).ToList();
        var causalOrder = new List<int>();
        var currentData = new Matrix<T>(n, d);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                currentData[i, j] = standardized[i, j];

        // Iteratively find root and regress out
        while (remaining.Count > 0)
        {
            // Find the variable with minimum total dependence on remaining variables
            int root = -1;
            double minDep = double.MaxValue;

            foreach (int j in remaining)
            {
                double totalDep = 0;
                var col = currentData.GetColumn(j);

                foreach (int k in remaining)
                {
                    if (k == j) continue;
                    var resid = RegressOut(col, currentData.GetColumn(k));

                    // Measure non-Gaussianity of residuals (kurtosis-based independence)
                    double kurtRes = Math.Abs(ComputeKurtosis(resid) - 3.0);
                    double kurtOrig = Math.Abs(ComputeKurtosis(col) - 3.0);
                    totalDep += Math.Abs(kurtOrig - kurtRes);
                }

                if (totalDep < minDep)
                {
                    minDep = totalDep;
                    root = j;
                }
            }

            if (root < 0) root = remaining[0];

            causalOrder.Add(root);
            remaining.Remove(root);

            // Regress out root from remaining variables
            if (remaining.Count > 0)
            {
                var rootCol = currentData.GetColumn(root);
                var newData = new Matrix<T>(n, d);
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < d; j++)
                        newData[i, j] = currentData[i, j];

                foreach (int j in remaining)
                {
                    var col = currentData.GetColumn(j);
                    var resid = RegressOut(col, rootCol);
                    for (int i = 0; i < n; i++) newData[i, j] = resid[i];
                }

                currentData = newData;
            }
        }

        // Estimate B matrix using OLS in causal order
        var B = new Matrix<T>(d, d);
        for (int idx = 1; idx < d; idx++)
        {
            int j = causalOrder[idx];
            var yCol = standardized.GetColumn(j);

            for (int pidx = 0; pidx < idx; pidx++)
            {
                int parent = causalOrder[pidx];
                var xCol = standardized.GetColumn(parent);

                // Simple regression coefficient
                T nT = NumOps.FromDouble(n);
                T mx = NumOps.Zero, my = NumOps.Zero;
                for (int i = 0; i < n; i++)
                {
                    mx = NumOps.Add(mx, xCol[i]);
                    my = NumOps.Add(my, yCol[i]);
                }
                mx = NumOps.Divide(mx, nT);
                my = NumOps.Divide(my, nT);

                T sxy = NumOps.Zero, sxx = NumOps.Zero;
                for (int i = 0; i < n; i++)
                {
                    T dx = NumOps.Subtract(xCol[i], mx);
                    sxy = NumOps.Add(sxy, NumOps.Multiply(dx, NumOps.Subtract(yCol[i], my)));
                    sxx = NumOps.Add(sxx, NumOps.Multiply(dx, dx));
                }

                double sxx_d = NumOps.ToDouble(sxx);
                T beta = sxx_d > 1e-10 ? NumOps.Divide(sxy, sxx) : NumOps.Zero;
                double beta_d = NumOps.ToDouble(beta);
                if (Math.Abs(beta_d) >= _threshold)
                    B[parent, j] = beta;
            }
        }

        return B;
    }
}
