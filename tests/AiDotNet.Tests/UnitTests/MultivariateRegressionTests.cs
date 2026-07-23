using AiDotNet.Models;
using AiDotNet.Regression;

namespace AiDotNetTests.UnitTests;

public class MultivariateRegressionTests
{
    private readonly double[][] _inputs = new double[][] { new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, new double[] { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 } };
    private readonly double[][] _outputs = new double[][] { new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, new double[] { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 } };
}
