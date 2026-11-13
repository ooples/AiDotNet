using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Regression;

/// <summary>
/// The Newton iteration of multinomial logistic regression against its own optimality condition.
/// </summary>
public class MultinomialLogisticRegressionNewtonTests
{
    /// <summary>
    /// At the maximum-likelihood fit the score equations hold: for every class c and feature j (intercept included),
    /// sum_i (y_ic - p_ic) x_ij = 0. The gradient and Hessian order parameters feature-major while the coefficients are
    /// class-major; when the Newton update was copied across without that transpose, three overlapping classes did not
    /// reach this point.
    /// </summary>
    [Fact]
    [Trait("category", "unit")]
    public void Newton_fit_satisfies_the_score_equations()
    {
        const int n = 90, classes = 3;
        var x = new Matrix<double>(n, 2);
        var y = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            int c = i % classes;
            // Overlapping clusters, so the maximum-likelihood estimate is finite.
            x[i, 0] = c + Math.Sin(1.7 * i);
            x[i, 1] = (0.5 * c) + Math.Cos(2.3 * i);
            y[i] = c;
        }

        var model = new MultinomialLogisticRegression<double>(
            new MultinomialLogisticRegressionOptions<double> { MaxIterations = 100, Tolerance = 1e-10 });
        model.Train(x, y);
        var probabilities = model.PredictProbabilities(x);

        for (int c = 0; c < classes; c++)
        {
            for (int j = 0; j <= 2; j++)
            {
                double score = 0;
                for (int i = 0; i < n; i++)
                {
                    double feature = j < 2 ? x[i, j] : 1.0;
                    double indicator = (int)y[i] == c ? 1.0 : 0.0;
                    score += (indicator - probabilities[i, c]) * feature;
                }

                Assert.True(Math.Abs(score) < 1e-6, $"score for class {c}, feature {j} is {score}");
            }
        }
    }
}
