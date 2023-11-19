namespace AiDotNet.Interfaces;

public abstract class IQuartile
{
    internal readonly int MinimumSize = 3;

    internal abstract (double q1Value, double q2Value, double q3Value) FindQuartiles(double[] inputs);

    internal virtual (double q1Value, double q2Value, double q3Value) FindQuartiles(double[][] inputs)
    {
        var flatList = new List<double>();
        foreach (var inputArray in inputs)
        {
            flatList.AddRange(inputArray);
        }

        return FindQuartiles(flatList.ToArray());
    }
}