using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Regularization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Regression;

public class LogisticRegressionTests : RegressionModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new LogisticRegression<double>();

    /// <summary>
    /// Logistic regression classifies, so it is trained on two class labels rather than on the
    /// unrestricted continuous target this harness generates.
    /// </summary>
    /// <remarks>
    /// The split is at the median, which keeps the classes balanced and keeps the label monotone in the
    /// original target, so the structural invariants still mean what they meant. This used to be
    /// unnecessary only because the model silently fitted ordinary least squares whenever the target was
    /// not binary; it now rejects such a target, which is what scikit-learn and statsmodels do.
    /// </remarks>
    protected override Vector<double> ToTarget(Vector<double> y) => ToT(ThresholdAtMedian(y));

    /// <summary>
    /// Predictions are probabilities in (0,1) from a logit link, not values on the target's own scale.
    /// </summary>
    /// <remarks>
    /// The equivariance and generic-quality invariants assume an additive identity-link estimator whose
    /// predictions live on the same scale as the response. A shifted or rescaled target produces the same
    /// median split and therefore the same probabilities, so translation and scaling equivariance cannot
    /// hold; and R-squared of a probability against a 0/1 label is not the quantity this model optimizes.
    /// </remarks>
    protected override bool IdentityLinkInvariantsApplicable => false;

    // Three features, 200 rows, a linear rule with an intercept and every seventh label flipped, so the classes
    // overlap and the unpenalized maximum-likelihood fit is finite.
    private static (Matrix<double> X, Vector<double> Y) OverlappingClasses()
    {
        const int n = 200, p = 3;
        var x = new Matrix<double>(n, p);
        var y = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
                x[i, j] = Math.Sin(0.37 * (i + 1) * (j + 1)) * (j + 1) + 0.1 * j;
            double z = 2.0 * x[i, 0] - 1.0 * x[i, 1] + 0.5 * x[i, 2] + 0.3;
            y[i] = z > 0 ? 1.0 : 0.0;
            if (i % 7 == 0) y[i] = 1.0 - y[i];
        }

        return (x, y);
    }

    [Fact]
    public void Train_MatchesScikitLearnUnpenalizedFit()
    {
        // scikit-learn 1.9.1: LogisticRegression(C=np.inf, tol=1e-12, max_iter=100000).fit(X, y) on the same rows
        // (L-BFGS, 13 iterations): coef [1.6043156922792248, -0.4229598633027594, 0.4277209006713329],
        // intercept 0.6320495715983193. The maximum-likelihood fit is unique, so any solver must reach it.
        var (x, y) = OverlappingClasses();
        var model = new LogisticRegression<double>(new LogisticRegressionOptions<double> { Tolerance = 1e-10 });

        model.Train(x, y);

        Assert.Equal(1.6043156922792248, model.Coefficients[0], 4);
        Assert.Equal(-0.4229598633027594, model.Coefficients[1], 4);
        Assert.Equal(0.4277209006713329, model.Coefficients[2], 4);
        Assert.Equal(0.6320495715983193, model.Intercept, 4);
    }

    [Fact]
    public void Train_GradientDescentSolverReachesTheSameRegularizedMinimum()
    {
        // Both solvers minimize one objective, so with L2 they must agree. The penalty used to enter gradient ASCENT
        // with the descent sign, pushing the coefficients away from zero during the fit; a penalty whose gradient
        // disagrees with its value would also send L-BFGS's line search and plain descent to different points.
        var (x, y) = OverlappingClasses();
        LogisticRegression<double> Fit(bool gradientDescent)
        {
            var options = new LogisticRegressionOptions<double> { Tolerance = 1e-9, MaxIterations = 200_000, LearningRate = 1.0 };
            var model = new LogisticRegression<double>(options,
                new L2Regularization<double, Matrix<double>, Vector<double>>(new RegularizationOptions { Strength = 0.1 }));
            if (gradientDescent) options.Solver = model.CreateGradientDescentSolver();
            model.Train(x, y);
            return model;
        }

        var lbfgs = Fit(gradientDescent: false);
        var descent = Fit(gradientDescent: true);
        var unpenalized = new LogisticRegression<double>(new LogisticRegressionOptions<double> { Tolerance = 1e-10 });
        unpenalized.Train(x, y);

        for (int j = 0; j < 3; j++)
            Assert.Equal(descent.Coefficients[j], lbfgs.Coefficients[j], 4);
        Assert.Equal(descent.Intercept, lbfgs.Intercept, 4);

        double Norm(Vector<double> v) => Math.Sqrt(v.DotProduct(v));
        Assert.True(Norm(lbfgs.Coefficients) < Norm(unpenalized.Coefficients),
            $"L2 should shrink the coefficients: penalized norm {Norm(lbfgs.Coefficients)}, unpenalized {Norm(unpenalized.Coefficients)}.");
    }
}
