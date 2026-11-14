using System.Reflection;
using AiDotNet.Interfaces;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// Coordinate descent must sweep the coordinates sequentially (Gauss-Seidel): the derivative for coordinate i is
/// taken at the solution with coordinates 0..i-1 already moved. Taking every derivative at the starting point is a
/// plain gradient step, which diverges on coupled objectives that the sequential sweep solves.
/// </summary>
public class CoordinateDescentSweepTests
{
    private const BindingFlags Private = BindingFlags.NonPublic | BindingFlags.Instance;

    [Fact]
    public void UpdateSolution_TakesEachDerivativeAfterThePreviousCoordinateMoved()
    {
        // Two identical features make the loss depend on a + b only, so the coordinates are coupled.
        var x = new Matrix<double>(4, 2);
        var y = new Vector<double>(4);
        double[] signs = { 1.0, -1.0, 1.0, -1.0 };
        for (int i = 0; i < 4; i++)
        {
            x[i, 0] = signs[i];
            x[i, 1] = signs[i];
            y[i] = 3.0 * signs[i];
        }

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = x, YTrain = y,
            XValidation = x, YValidation = y,
            XTest = x, YTest = y
        };

        const double learningRate = 0.75;
        var start = new MultipleRegression<double>(new RegressionOptions<double> { UseIntercept = false })
            .WithParameters(new Vector<double>(new[] { 0.0, 0.0 }));
        var optimizer = new CoordinateDescentOptimizer<double, Matrix<double>, Vector<double>>(
            start,
            new CoordinateDescentOptimizerOptions<double, Matrix<double>, Vector<double>>
            {
                InitialLearningRate = learningRate,
                InitialMomentum = 0.0,
                MaxIterations = 1
            });

        var type = optimizer.GetType();
        var initialize = type.GetMethod("InitializeAdaptiveParameters", Private, null,
            new[] { typeof(IFullModel<double, Matrix<double>, Vector<double>>) }, null);
        var derivative = type.GetMethod("CalculatePartialDerivative", Private);
        // By signature: the base class's protected UpdateSolution(model, Vector) overload is visible to these flags too.
        var update = type.GetMethod("UpdateSolution", Private, null,
            new[] { typeof(IFullModel<double, Matrix<double>, Vector<double>>), typeof(OptimizationInputData<double, Matrix<double>, Vector<double>>) }, null);
        Assert.NotNull(initialize);
        Assert.NotNull(derivative);
        Assert.NotNull(update);
        initialize.Invoke(optimizer, new object[] { start });

        // Expected Gauss-Seidel sweep, built from the optimizer's own derivative estimate.
        double d0 = Assert.IsType<double>(derivative.Invoke(optimizer, new object[] { start, inputData, 0 }));
        double a = 0.0 - learningRate * d0;
        var afterFirst = start.WithParameters(new Vector<double>(new[] { a, 0.0 }));
        double d1 = Assert.IsType<double>(derivative.Invoke(optimizer, new object[] { afterFirst, inputData, 1 }));
        double b = 0.0 - learningRate * d1;

        // The simultaneous (gradient-step) b would use the derivative at the starting point.
        double d1AtStart = Assert.IsType<double>(derivative.Invoke(optimizer, new object[] { start, inputData, 1 }));
        double simultaneousB = 0.0 - learningRate * d1AtStart;
        Assert.True(Math.Abs(b - simultaneousB) > 1e-3, "The fixture must couple the coordinates.");

        var result = Assert.IsAssignableFrom<IFullModel<double, Matrix<double>, Vector<double>>>(
            update.Invoke(optimizer, new object[] { start, inputData }));
        var parameters = result.GetParameters();

        Assert.Equal(a, parameters[0], 1e-9);
        Assert.Equal(b, parameters[1], 1e-9);
    }
}
