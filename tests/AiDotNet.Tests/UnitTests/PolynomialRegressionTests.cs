using AiDotNet.Regression;

namespace AiDotNetTests.UnitTests;

public class PolynomialRegressionTests
{
    private readonly double[] _inputs = new double[] { 171, 183, 12, 261, 77, 272, 36, 6, 213, 4, 74, 79, 158, 60, 24, 99, 292, 30, 176, 276, 285, 38, 64, 21, 37, 258, 141, 46, 48, 128, 165, 74, 102, 6, 53, 23, 56, 236, 104, 96, 228, 216, 116, 160, 38, 106, };
    private readonly double[] _outputs = new double[] { 144, 87, 216, 111, 49, 300, 96, 138, 165, 164, 62, 60, 31, 324, 368, 76, 246, 138, 57, 76, 66, 116, 128, 4, 130, 73, 372, 73, 12, 16, 20, 46, 7, 280, 106, 27, 35, 126, 100, 91, 156, 14, 14, 48, 81, 75, };
}
