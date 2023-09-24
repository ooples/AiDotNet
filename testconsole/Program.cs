using AiDotNet.Normalization;
using AiDotNet.Regression;

namespace AiDotNetTestConsole
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var inputs = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            var outputs = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

            var simpleRegression = new SimpleRegression(inputs, outputs);
            var metrics1 = simpleRegression.Metrics;
            var predictions1 = simpleRegression.Predictions;

            var advancedSimpleRegression = new SimpleRegression(inputs, outputs, trainingPctSize: 20, normalization: new LogNormalization());
            var metrics2 = advancedSimpleRegression.Metrics;
            var predictions2 = advancedSimpleRegression.Predictions;

            Console.WriteLine();
        }
    }
}