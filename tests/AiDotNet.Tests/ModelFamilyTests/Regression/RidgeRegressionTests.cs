using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Regression;

public class RidgeRegressionTests : RegressionModelTestBase
{
    protected override IFullModel<double, Matrix<double>, Vector<double>> CreateModel()
        => new RidgeRegression<double>();

    /// <summary>
    /// double and float build the normal equations in one blocked pass; every other type appends the intercept column
    /// and multiplies. Both must give the same fit, with and without an intercept, over more rows than one block.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Train_DoublePassMatchesGenericNormalEquations(bool useIntercept)
    {
        const int rows = 2500, columns = 4;
        var xDouble = new Matrix<double>(rows, columns);
        var xDecimal = new Matrix<decimal>(rows, columns);
        var yDouble = new Vector<double>(rows);
        var yDecimal = new Vector<decimal>(rows);
        for (int i = 0; i < rows; i++)
        {
            double target = 3.0;
            for (int j = 0; j < columns; j++)
            {
                // Exact in both types: multiples of 1/8.
                double value = ((i * (j + 3) + 5 * j) % 41 - 20) / 8.0;
                xDouble[i, j] = value;
                xDecimal[i, j] = (decimal)value;
                target += (j + 1) * value;
            }

            target += (i % 5 - 2) / 4.0;
            yDouble[i] = target;
            yDecimal[i] = (decimal)target;
        }

        var fast = new RidgeRegression<double>(new RidgeRegressionOptions<double> { Alpha = 2.0, UseIntercept = useIntercept });
        var generic = new RidgeRegression<decimal>(new RidgeRegressionOptions<decimal> { Alpha = 2.0, UseIntercept = useIntercept });
        fast.Train(xDouble, yDouble);
        generic.Train(xDecimal, yDecimal);

        for (int j = 0; j < columns; j++)
            Assert.Equal((double)generic.Coefficients[j], fast.Coefficients[j], 9);
        Assert.Equal((double)generic.Intercept, fast.Intercept, 9);
        if (useIntercept)
            Assert.True(Math.Abs(fast.Intercept) > 1.0, $"The fit should find the intercept near 3; got {fast.Intercept}.");
    }
}
