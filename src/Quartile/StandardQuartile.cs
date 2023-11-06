namespace AiDotNet.Quartile;

/// <summary>
/// Performs a simple Quartile statistical calculation using the standard calculation method.
/// Q1 = 25% Q2 = 50% or Median and Q3 = 75% of the data.
/// </summary>
/// <param name="inputs">The input array that the Quartile method will be run on. Must be sorted in ascending order.</param>
/// <exception cref="ArgumentException">The input array does not have enough data.</exception>
public class StandardQuartile : IQuartile
{
    internal override (double q1Value, double q2Value, double q3Value) FindQuartiles(double[] inputs)
    {
        var nSize = inputs.Length;
        var isOdd = ((double)nSize).IsOdd();
        ValidationHelper.CheckForMinimumInputSize(nSize, MinimumSize);

        //Assume inputs array is sorted.

        //Wolfram Alpha Method
        var nQ1 = (int)(nSize / 4.0);
        var nQ2 = (int)(nSize / 2.0);
        var nQ3 = (int)(3 * nSize / 4.0);

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