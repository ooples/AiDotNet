using AiDotNet.Classification.Ordinal;
using System.Reflection;
using System.Threading.Tasks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Classification;

/// <summary>
/// Checks the analytic negative log-likelihood gradient of the cumulative logit model
/// (McCullagh, 1980) against central finite differences.
/// </summary>
public class OrdinalLogisticRegressionGradientTests
{
    private static readonly double[] Coefficients = { 0.7, -0.4 };
    private static readonly double[] Thresholds = { -0.5, 0.8 };

    [Fact(Timeout = 120000)]
    public async Task ComputeGradients_MatchesFiniteDifferencesOfMeanNegativeLogLikelihood()
    {
        // xunit only honours Timeout on async tests
        await Task.Yield();

        // Arrange: three ordered classes so the middle class contributes
        var x = new Matrix<double>(9, 2);
        var y = new Vector<double>(9);
        double[,] features =
        {
            { -1.0, 0.5 }, { -0.6, -0.3 }, { -0.2, 1.1 },
            { 0.1, -0.7 }, { 0.4, 0.2 }, { 0.3, -1.2 },
            { 1.2, 0.4 }, { 0.9, -0.5 }, { 1.5, 1.0 }
        };
        double[] labels = { 0, 0, 1, 1, 1, 1, 2, 2, 2 };
        for (int i = 0; i < 9; i++)
        {
            x[i, 0] = features[i, 0];
            x[i, 1] = features[i, 1];
            y[i] = labels[i];
        }

        var trained = new OrdinalLogisticRegression<double>(maxIterations: 5);
        trained.Train(x, y);
        var parameters = new Vector<double>(new[] { Coefficients[0], Coefficients[1], Thresholds[0], Thresholds[1] });

        // The fitted state is set directly. SetParameters / WithParameters cannot be used here: the parameter
        // manifest holds both the generated component for the coefficient vector and the packed
        // coefficients-plus-thresholds component, so a 4-value restore is split 2 + 2 and the packed half is
        // unpacked as one coefficient and one threshold.
        SetField(trained, "_coefficients", new Vector<double>(Coefficients));
        SetField(trained, "_thresholds", new Vector<double>(Thresholds));
        var model = trained;

        // Act
        var gradient = model.ComputeGradients(x, y);

        // Assert
        Assert.Equal(4, gradient.Length);
        const double h = 1e-6;
        for (int j = 0; j < 4; j++)
        {
            var plus = parameters.ToArray();
            var minus = parameters.ToArray();
            plus[j] += h;
            minus[j] -= h;
            double numeric = (MeanNegativeLogLikelihood(plus, x, labels) - MeanNegativeLogLikelihood(minus, x, labels)) / (2 * h);
            Assert.Equal(numeric, gradient[j], 6);
        }
    }

    /// <summary>
    /// Mean of -log P(Y = y) with P(Y &lt;= k) = sigmoid(theta_k - beta . x).
    /// </summary>
    private static double MeanNegativeLogLikelihood(double[] parameters, Matrix<double> x, double[] labels)
    {
        double total = 0;
        for (int i = 0; i < x.Rows; i++)
        {
            double eta = parameters[0] * x[i, 0] + parameters[1] * x[i, 1];
            double c0 = Sigmoid(parameters[2] - eta);
            double c1 = Sigmoid(parameters[3] - eta);
            int label = (int)labels[i];
            double p = label == 0 ? c0 : label == 1 ? c1 - c0 : 1 - c1;
            total -= Math.Log(p);
        }

        return total / x.Rows;
    }

    private static void SetField(object target, string name, object value)
    {
        for (var type = target.GetType(); type is not null; type = type.BaseType)
        {
            var field = type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.DeclaredOnly);
            if (field is not null)
            {
                field.SetValue(target, value);
                return;
            }
        }

        Assert.Fail($"Field {name} not found on {target.GetType().Name}.");
    }

    private static double Sigmoid(double z) => 1.0 / (1.0 + Math.Exp(-z));
}
