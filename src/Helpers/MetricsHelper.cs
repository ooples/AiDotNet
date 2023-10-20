namespace AiDotNet.Helpers;

internal static class MetricsHelper
{
    internal static (double residualSquaresSum, double totalSquaresSum, double r2) CalculateR2(double[] inputs, double[] outputs, double outputsAvg, int sampleSize)
    {
        double residualSumSquares = 0, totalSumSquares = 0;
        for (var i = 0; i < sampleSize; i++)
        {
            residualSumSquares += Math.Pow(outputs[i] - inputs[i], 2);
            totalSumSquares += Math.Pow(outputs[i] - outputsAvg, 2);
        }

        return (residualSumSquares, totalSumSquares, totalSumSquares != 0 ? 1 - (residualSumSquares / totalSumSquares) : 0);
    }

    internal static (double residualSquaresSum, double totalSquaresSum, double r2) CalculateR2(double[][] inputs, double[][] outputs, double outputsAvg, int sampleSize)
    {
        double residualSumSquares = 0, totalSumSquares = 0;
        for (var i = 0; i < sampleSize; i++)
        {
            for (var j = 0; j < inputs[0].Length; j++)
            {
                residualSumSquares += Math.Pow(outputs[i][j] - inputs[i][j], 2);
                totalSumSquares += Math.Pow(outputs[i][j] - outputsAvg, 2);
            }
        }

        return (residualSumSquares, totalSumSquares, totalSumSquares != 0 ? 1 - (residualSumSquares / totalSumSquares) : 0);
    }

    internal static double Average(this double[][] valuesArray)
    {
        double sum = 0;
        var count = 0;
        for (var i = 0; i < valuesArray.Length; i++)
        {
            for (var j = 0; j < valuesArray[i].Length; j++)
            {
                sum += valuesArray[i][j];
                count++;
            }
        }

        return sum / count;
    }
}