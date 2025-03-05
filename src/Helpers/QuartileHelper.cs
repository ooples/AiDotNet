namespace AiDotNet.Helpers
{
    internal static class QuartileHelper
    {
        internal static bool IsOdd(this double value) => value % 2 != 0;

        internal static (double q1Value, double q2Value, double q3Value) FindQuartiles(double[] inputs, IQuartile quartile)
        {
            return quartile.FindQuartiles(inputs);
        }

        internal static (double[], double[]) FilterArraysWithIndices(IEnumerable<double> rawInputs, IEnumerable<double> rawOutputs, IEnumerable<int> ignoredIndices)
        {
            return ([.. rawInputs.Where((val, idx) => !ignoredIndices.Contains(idx))],
                [.. rawOutputs.Where((val, idx) => !ignoredIndices.Contains(idx))]);
        }

        internal static int[] FindIndicesToRemove(double[] unfiltered, double minLimit, double maxLimit)
        {
            var indicesToRemove = new List<int>();
            //Determine Indices to filter based on minLimit and maxLimit
            for (var i = 0; i < unfiltered.Length; i++)
            {
                if (unfiltered[i] >= minLimit) { break; }

                indicesToRemove.Add(i);
            }
            for (var y = unfiltered.Length - 1; y >= 0; y--)
            {
                if (unfiltered[y] <= maxLimit) { break; }

                indicesToRemove.Add(y);
            }

            return indicesToRemove.ToArray();
        }
    }
}
