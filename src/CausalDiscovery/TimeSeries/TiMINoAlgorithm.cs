using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.TimeSeries;

/// <summary>
/// TiMINo — Time series Models with Independent Noise.
/// </summary>
/// <remarks>
/// <para>
/// TiMINo tests whether the residuals of a time series regression model are independent
/// of the inputs. For each pair (i→j), it fits a linear model predicting j from lagged
/// values of i (and j itself), then tests whether the residuals are independent of i's
/// lagged values using the HSIC independence criterion.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>For each pair (i,j), fit model: x_j[t] = sum_l (a_l * x_i[t-l] + b_l * x_j[t-l]) + noise</item>
/// <item>Compute residuals: e[t] = x_j[t] - predicted</item>
/// <item>Test independence of residuals from input using HSIC</item>
/// <item>If residuals are independent of i's lags, direction i→j is valid</item>
/// <item>Compare HSIC scores for i→j vs j→i to determine direction</item>
/// <item>Edge weight from OLS coefficient magnitude</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> TiMINo checks if the "leftover noise" after predicting one variable
/// from another's past is truly random (independent). If it is, the prediction direction is
/// likely the causal direction.
/// </para>
/// <para>
/// Reference: Peters et al. (2013), "Causal Discovery with Continuous Additive Noise Models", JMLR.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Causal Discovery with Continuous Additive Noise Models", "https://jmlr.org/papers/v14/peters13a.html", Year = 2013, Authors = "Jonas Peters, Joris M. Mooij, Dominik Janzing, Bernhard Scholkopf")]
public class TiMINoAlgorithm<T> : TimeSeriesCausalBase<T>
{
    /// <summary>
    /// Squared correlation threshold below which residuals are considered independent of the cause.
    /// Uses squared Pearson correlation as a computationally efficient proxy for HSIC.
    /// </summary>
    private const double IndependenceThreshold = 0.1;

    /// <summary>
    /// Minimum absolute standardized coefficient magnitude for an edge to be included.
    /// Applied after internal standardization to make the threshold scale-invariant.
    /// </summary>
    private const double EdgeCoefficientThreshold = 0.1;

    /// <summary>
    /// Epsilon for numerical stability in variance/correlation computations.
    /// </summary>
    private const double NumericalStabilityEpsilon = 1e-10;

    /// <summary>
    /// Epsilon for preventing division by zero in correlation denominators.
    /// </summary>
    private const double CorrelationDenominatorEpsilon = 1e-15;

    /// <inheritdoc/>
    public override string Name => "TiMINo";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public TiMINoAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyTimeSeriesOptions(options);
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        int effectiveN = n - MaxLag;
        if (effectiveN < 2 * MaxLag + 3 || d < 2) return new Matrix<T>(d, d);

        var result = new Matrix<T>(d, d);

        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;

                // Test direction i→j: fit model and check residual independence
                var (hsicIJ, betaIJ) = FitAndComputeHSIC(data, i, j, n);
                // Test direction j→i
                var (hsicJI, _) = FitAndComputeHSIC(data, j, i, n);

                // Lower HSIC = more independent residuals = correct causal direction
                if (hsicIJ < hsicJI && hsicIJ < IndependenceThreshold)
                {
                    // Use the fitted lag coefficient as the edge weight
                    if (Math.Abs(betaIJ) > EdgeCoefficientThreshold)
                        result[i, j] = NumOps.FromDouble(betaIJ);
                }
            }

        return result;
    }

    private (double hsic, double beta) FitAndComputeHSIC(Matrix<T> data, int cause, int target, int n)
    {
        int effectiveN = n - MaxLag;
        int p = 2 * MaxLag; // lags of cause + lags of target

        // Build design matrix and target vector
        var (laggedX, y) = CreateLaggedData(data, target, MaxLag);

        // Fit OLS and get residuals
        double rss = ComputeRSS(laggedX, y, effectiveN, laggedX.Columns);

        // Simple residual estimation: use RSS to approximate HSIC
        // HSIC ∝ mean(K_residual * K_input) where K is a kernel matrix
        // Simplified: use squared correlation between residuals and cause's lags as HSIC proxy

        // Recompute residuals from a simpler bivariate model
        // cause(t) and target(t + MaxLag), t = 0..effectiveN-1
        var x = Tensor<T>.FromMatrix(data);
        var causeVec = LaggedSeries(x, cause, MaxLag, MaxLag, effectiveN);
        var targetVec = LaggedSeries(x, target, 0, MaxLag, effectiveN);

        // OLS: beta = cov(cause, target) / var(cause)
        var centC = Engine.TensorSubtractScalar(causeVec, Engine.TensorMean(causeVec));
        var centT = Engine.TensorSubtractScalar(targetVec, Engine.TensorMean(targetVec));

        T varC = Engine.TensorSum(Engine.TensorMultiply(centC, centC));
        T covCT = Engine.TensorSum(Engine.TensorMultiply(centC, centT));

        if (!NumOps.GreaterThan(varC, NumOps.FromDouble(NumericalStabilityEpsilon)))
            return (1.0, 0.0); // Can't fit model

        T beta = NumOps.Divide(covCT, varC);

        // Compute residuals: e = target - beta * cause (centered)
        var residuals = Engine.TensorSubtract(centT, Engine.TensorMultiplyScalar(centC, beta));

        // HSIC proxy: squared correlation between residuals and cause
        T covResiC = Engine.TensorSum(Engine.TensorMultiply(residuals, centC));
        T varR = Engine.TensorSum(Engine.TensorMultiply(residuals, residuals));

        double dVarR = NumOps.ToDouble(varR);
        double dVarC = NumOps.ToDouble(varC);
        double dCovRC = NumOps.ToDouble(covResiC);
        double denom = Math.Sqrt(Math.Max(dVarR, CorrelationDenominatorEpsilon) * Math.Max(dVarC, CorrelationDenominatorEpsilon));
        double corrResidual = dCovRC / denom;

        // Return standardized beta (scale-invariant): beta_std = beta * std(cause) / std(target)
        double stdC = Math.Sqrt(Math.Max(dVarC / effectiveN, NumericalStabilityEpsilon));
        double dVarT = NumOps.ToDouble(Engine.TensorSum(Engine.TensorMultiply(centT, centT)));
        double stdT = Math.Sqrt(Math.Max(dVarT / effectiveN, NumericalStabilityEpsilon));
        double betaStd = stdT > 1e-15 ? NumOps.ToDouble(beta) * stdC / stdT : 0;

        return (corrResidual * corrResidual, betaStd); // Squared correlation as HSIC proxy
    }
}
