namespace AiDotNet;

public class Metrics : IMetrics
{
    public double MeanSquaredError { get; private set; }
    public double RootMeanSquaredError { get; private set; }

    private double[] _oosPredictions { get; }
    private double[] _oosActualValues { get; }

    public Metrics(double[] OosPredictions, double[] OosActualValues)
    {
        _oosPredictions = OosPredictions;
        _oosActualValues = OosActualValues;

        MeanSquaredError = CalculateMeanSquaredError();
        RootMeanSquaredError = CalculateRootMeanSquaredError();
    }

    internal sealed override double CalculateMeanSquaredError()
    {
        double sum = 0;
        for (var i = 0; i < _oosPredictions.Length; i++)
        {
            sum += Math.Pow(_oosActualValues[i] - _oosPredictions[i], 2);
        }

        return sum / _oosPredictions.Length;
    }

    internal sealed override double CalculateRootMeanSquaredError()
    {
        return MeanSquaredError >= 0 ? Math.Sqrt(MeanSquaredError) : 0;
    }
}