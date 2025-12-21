using AiDotNet.Models;
using AiDotNet.Regression;

namespace AiDotNetTests.UnitTests;

public class WeightedRegressionTests
{
    private readonly double[] _inputs = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    private readonly double[] _outputs = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    private readonly double[] _weights = new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
}
