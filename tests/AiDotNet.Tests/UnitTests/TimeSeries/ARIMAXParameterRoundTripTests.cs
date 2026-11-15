using System;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// ARIMAX packs and reads its parameters in one order ([AR..., MA..., exogenous..., intercept]), so a
/// parameter vector survives every get/set/with path and leaves the forecast unchanged.
/// </summary>
public class ARIMAXParameterRoundTripTests
{
    private const int N = 40;

    private static (Matrix<double> X, Vector<double> Y) Data()
    {
        var x = new Matrix<double>(N, 1);
        var y = new Vector<double>(N);
        double previous = 0.0;
        for (int i = 0; i < N; i++)
        {
            double exogenous = Math.Sin(0.3 * i) + 0.05 * i;
            x[i, 0] = exogenous;
            // Deterministic wobble in place of noise
            double value = 2.0 + 0.5 * exogenous + 0.3 * previous + 0.1 * Math.Cos(1.7 * i);
            y[i] = value;
            previous = value;
        }

        return (x, y);
    }

    private static ARIMAXModel<double> Trained()
    {
        var model = new ARIMAXModel<double>(new ARIMAXModelOptions<double>
        {
            AROrder = 1,
            MAOrder = 1,
            DifferenceOrder = 0,
            ExogenousVariables = 1
        });
        var (x, y) = Data();
        model.Train(x, y);
        return model;
    }

    private static void AssertSame(Vector<double> expected, Vector<double> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i]);
        }
    }

    [Fact]
    public void WithParameters_OfOwnParameters_LeavesForecastUnchanged()
    {
        var model = Trained();
        var (x, _) = Data();
        var parameters = model.GetParameters();
        var before = model.Predict(x);

        var copy = model.WithParameters(parameters);

        AssertSame(parameters, copy.GetParameters());
        AssertSame(before, copy.Predict(x));
    }

    [Fact]
    public void SetParameters_OfOwnParameters_LeavesForecastUnchanged()
    {
        var model = Trained();
        var (x, _) = Data();
        var parameters = model.GetParameters();
        var before = model.Predict(x);

        model.SetParameters(parameters);

        AssertSame(parameters, model.GetParameters());
        AssertSame(before, model.Predict(x));
    }

    [Fact]
    public void DistinctParameters_RoundTripThroughSetAndWith_InOneOrder()
    {
        var model = Trained();
        var (x, _) = Data();
        int count = model.GetParameters().Length;

        // AR, MA, exogenous coefficients and the intercept: 1 + 1 + 1 + 1
        Assert.Equal(4, count);

        // Distinct values so any reordering is visible: AR 0.1, MA 0.2, exogenous 0.3, intercept 0.4
        var distinct = new Vector<double>(new[] { 0.1, 0.2, 0.3, 0.4 });

        var viaWith = model.WithParameters(distinct);
        AssertSame(distinct, viaWith.GetParameters());

        model.SetParameters(distinct);
        AssertSame(distinct, model.GetParameters());
        Assert.Equal(0.4, Convert.ToDouble(model.GetModelMetadata().AdditionalInfo["Intercept"]));

        // Both paths hold the same coefficients, so they forecast the same values
        AssertSame(model.Predict(x), viaWith.Predict(x));
    }
}
