namespace AiDotNet.Quartile;

public class InclusiveQuartile : IQuartile
{
    /// <summary>
    /// Performs a simple Quartile statistical calculation using the Inclusive calculation method.
    /// Inclusive method includes the Median.
    /// Q1 = 25% Q2 = 50% or Median and Q3 = 75% of the data.
    /// </summary>
    /// <param name="inputs">The input array that the Quartile method will be run on. Must be sorted in ascending order.</param>
    /// <exception cref="ArgumentException">The input array does not have enough data.</exception>
    internal override (double q1Value, double q2Value, double q3Value) FindQuartiles(double[] inputs)
    {
        var nSize = inputs.Length;
        var isOdd = ((double)nSize).IsOdd();
        ValidationHelper.CheckForMinimumInputSize(nSize, MinimumSize);

        //Assume inputs array is sorted.

        //Inclusive Method (Counts the Median)
        var nQ1 = (int)((nSize + 3)/ 4.0);
        var nQ2 = (int)((nSize + 1)/ 2.0);
        var nQ3 = (int)((3 * nSize + 1)/ 4.0);

        var valQ1 = isOdd ? inputs[nQ1] : (inputs[nQ1] + inputs[nQ1 - 1]) / 2;
        var valQ2 = isOdd ? inputs[nQ2] : (inputs[nQ2] + inputs[nQ2 - 1]) / 2;
        var valQ3 = isOdd ? inputs[nQ3] : (inputs[nQ3] + inputs[nQ3 - 1]) / 2;

        return (valQ1, valQ2, valQ3);
    }

    internal override (double q1Value, double q2Value, double q3Value) FindQuartiles(double[][] inputs)
    {
        throw new NotImplementedException();
    }
}