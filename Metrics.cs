namespace AiDotNet;

public class Metrics : IMetrics
{
    public double MeanSquaredError { get; private set; }

    private double[] _oosPredictions { get; }
    private double[] _oosActualValues { get; }

    public Metrics(double[] OosPredictions, double[] OosActualValues)
    {
        _oosPredictions = OosPredictions;
        _oosActualValues = OosActualValues;
    }

    public double CalculateMeanSquaredError()
    {
        double sum = 0;
        for (var i = 0; i < _oosPredictions.Length; i++)
        {
            sum += Math.Pow(_oosPredictions[i] - _oosActualValues[i], 2);
        }
        MeanSquaredError = sum / _oosPredictions.Length;

        return MeanSquaredError;
    }
}