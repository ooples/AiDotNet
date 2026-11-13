using System;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// Reference checks for the VAR model against statsmodels 0.15 (<c>VAR(y).fit(2, trend='c')</c>).
/// </summary>
public class VectorAutoRegressionReferenceTests
{
    private const double Tolerance = 1e-8;

    // t = 0..39: [sin(0.7 t) + 0.3 cos(1.3 t), cos(0.4 t) + 0.2 sin(2.1 t) + 0.05 t].
    private static Matrix<double> Series()
    {
        var y = new Matrix<double>(40, 2);
        for (int t = 0; t < 40; t++)
        {
            y[t, 0] = Math.Sin(0.7 * t) + (0.3 * Math.Cos(1.3 * t));
            y[t, 1] = Math.Cos(0.4 * t) + (0.2 * Math.Sin(2.1 * t)) + (0.05 * t);
        }

        return y;
    }

    private static VectorAutoRegressionModel<double> Fit(Matrix<double> y)
    {
        var model = new VectorAutoRegressionModel<double>(new VARModelOptions<double> { Lag = 2, OutputDimension = 2 });
        model.Train(y, y.GetColumn(0));
        return model;
    }

    [Fact]
    [Trait("category", "unit")]
    public void Coefficients_and_forecast_match_statsmodels()
    {
        var y = Series();
        var model = Fit(y);

        // r.coefs[0] | r.coefs[1], row per equation.
        double[,] expected =
        {
            { 1.4166971085511717, 0.00652226717178794, -0.9352805294493948, -0.00678045016481055 },
            { 0.01702248699299047, 1.228955240832602, -0.00358287252656803, -0.34305165602578097 }
        };
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                Assert.Equal(expected[i, j], model.Coefficients[i, j], Tolerance);
            }
        }

        // r.forecast(y[-2:], 2)
        var history = new Matrix<double>(2, 2);
        for (int j = 0; j < 2; j++)
        {
            history[0, j] = y[38, j];
            history[1, j] = y[39, j];
        }

        var forecast = model.Forecast(history, 2);
        Assert.Equal(0.4553146710183491, forecast[0, 0], Tolerance);
        Assert.Equal(1.0647136441375573, forecast[0, 1], Tolerance);
        Assert.Equal(-0.37403720034420407, forecast[1, 0], Tolerance);
        Assert.Equal(1.0785510359825268, forecast[1, 1], Tolerance);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Orthogonalized_impulse_responses_of_a_var2_match_statsmodels()
    {
        // r.irf(3).orth_irfs[:, :, shock]: rows are periods 0..3, columns the responding variables. A VAR(2) used to
        // throw (a k x k identity times the 2k x 2k companion matrix).
        var responses = Fit(Series()).ImpulseResponseAnalysis(3);

        double[,] shock0 =
        {
            { 2.1522932140272188e-01, 1.8244039246827079e-04 },
            { 3.0491594723164950e-01, 3.8879494005514370e-03 },
            { 2.3069786833639946e-01, 9.1348178362778594e-03 },
            { 4.1680272132055218e-02, 1.2727091266359437e-02 }
        };
        double[,] shock1 =
        {
            { 0.0, 0.37073342861712677 },
            { 0.00241802247095387, 0.4556147900508573 },
            { 0.0038835072939741, 0.43279062815868635 },
            { 0.00297372694556264, 0.37563834590818546 }
        };

        for (int h = 0; h < 4; h++)
        {
            for (int v = 0; v < 2; v++)
            {
                Assert.Equal(shock0[h, v], responses["Variable_0"][h, v], Tolerance);
                Assert.Equal(shock1[h, v], responses["Variable_1"][h, v], Tolerance);
            }
        }
    }
}
