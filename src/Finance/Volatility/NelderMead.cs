namespace AiDotNet.Finance.Volatility;

/// <summary>
/// Compact derivative-free Nelder–Mead (downhill simplex) minimizer for low-dimensional objectives — used
/// to maximize the Gaussian quasi-log-likelihood of the GARCH-family models over their (unconstrained,
/// via transform) parameters. This is the standard estimation route in econometric volatility packages
/// where closed-form gradients are awkward.
/// </summary>
internal static class NelderMead
{
    /// <summary>Minimizes <paramref name="f"/> starting from <paramref name="start"/> (length = <paramref name="dim"/>).</summary>
    public static double[] Minimize(Func<double[], double> f, double[] start, int dim,
        int maxIterations = 4000, double tolerance = 1e-10)
    {
        // Build the initial simplex: start + a perturbed vertex per dimension.
        int n = dim;
        var simplex = new double[n + 1][];
        var fval = new double[n + 1];
        simplex[0] = (double[])start.Clone();
        for (int i = 0; i < n; i++)
        {
            var v = (double[])start.Clone();
            double step = Math.Abs(v[i]) > 1e-6 ? 0.1 * v[i] : 0.1;
            v[i] += step;
            simplex[i + 1] = v;
        }

        for (int i = 0; i <= n; i++)
        {
            fval[i] = f(simplex[i]);
        }

        const double alpha = 1.0, gamma = 2.0, rho = 0.5, sigma = 0.5;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Order vertices by objective value (best first).
            Array.Sort(fval, simplex);

            // Convergence: spread of objective values is tiny.
            if (Math.Abs(fval[n] - fval[0]) <= tolerance * (Math.Abs(fval[0]) + tolerance))
            {
                break;
            }

            // Centroid of all but the worst vertex.
            var centroid = new double[n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++) centroid[j] += simplex[i][j];
            }

            for (int j = 0; j < n; j++) centroid[j] /= n;

            // Reflection.
            var reflected = Combine(centroid, simplex[n], alpha);
            double fr = f(reflected);
            if (fr < fval[0])
            {
                // Expansion.
                var expanded = Combine(centroid, simplex[n], gamma);
                double fe = f(expanded);
                Replace(simplex, fval, n, fe < fr ? expanded : reflected, Math.Min(fe, fr));
            }
            else if (fr < fval[n - 1])
            {
                Replace(simplex, fval, n, reflected, fr);
            }
            else
            {
                // Contraction toward the better of worst/reflected.
                bool outside = fr < fval[n];
                var basis = outside ? reflected : simplex[n];
                double fbasis = outside ? fr : fval[n];
                var contracted = new double[n];
                for (int j = 0; j < n; j++) contracted[j] = centroid[j] + rho * (basis[j] - centroid[j]);
                double fc = f(contracted);
                if (fc < fbasis)
                {
                    Replace(simplex, fval, n, contracted, fc);
                }
                else
                {
                    // Shrink toward the best vertex.
                    for (int i = 1; i <= n; i++)
                    {
                        for (int j = 0; j < n; j++)
                            simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]);
                        fval[i] = f(simplex[i]);
                    }
                }
            }
        }

        Array.Sort(fval, simplex);
        return simplex[0];
    }

    private static double[] Combine(double[] centroid, double[] worst, double coeff)
    {
        var result = new double[centroid.Length];
        for (int j = 0; j < centroid.Length; j++)
        {
            result[j] = centroid[j] + coeff * (centroid[j] - worst[j]);
        }

        return result;
    }

    private static void Replace(double[][] simplex, double[] fval, int worstIdx, double[] point, double value)
    {
        simplex[worstIdx] = point;
        fval[worstIdx] = value;
    }
}
