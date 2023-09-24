using AiDotNet.Interfaces;

namespace AiDotNet;

public sealed class Metrics : IMetrics
{
    public double MeanSquaredError { get; private set; }
    public double RootMeanSquaredError { get; private set; }
    public double AdjustedR2 { get; private set; }
    public double R2 { get; private set; }
    public double PredictionsStandardError { get; private set; }
    public double PredictionsStandardDeviation { get; private set; }
    public double AverageStandardError { get; private set; }
    public double AverageStandardDeviation { get; private set; }
    public int DegreesOfFreedom { get; private set; }
    
    private double[] _oosPredictions { get; }
    private double _oosPredictionsAvg { get; }
    private double[] _oosActualValues { get; }
    private double _oosActualValuesAvg { get; }
    private int _paramsCount { get; }
    private int _sampleSize { get; }
    private double _residualSumOfSquares { get; set; }
    private double _totalSumOfSquares { get; set; }

    public Metrics(double[] OosPredictions, double[] OosActualValues, int paramCount)
    {
        _oosPredictions = OosPredictions;
        _oosPredictionsAvg = _oosPredictions.Average();
        _oosActualValues = OosActualValues;
        _oosActualValuesAvg = _oosActualValues.Average();
        _paramsCount = paramCount;
        _sampleSize = _oosPredictions.Length;

        DegreesOfFreedom = CalculateDegreesOfFreedom();
        R2 = CalculateR2();
        AdjustedR2 = CalculateAdjustedR2(R2);
        AverageStandardDeviation = CalculateAverageStandardDeviation();
        PredictionsStandardDeviation = CalculatePredictionStandardDeviation();
        AverageStandardError = CalculateAverageStandardError();
        PredictionsStandardError = CalculatePredictionStandardError();
        MeanSquaredError = CalculateMeanSquaredError();
        RootMeanSquaredError = CalculateRootMeanSquaredError();
    }

    internal override double CalculateMeanSquaredError()
    {
        return _residualSumOfSquares / _sampleSize;
    }

    internal override double CalculateRootMeanSquaredError()
    {
        return MeanSquaredError >= 0 ? Math.Sqrt(MeanSquaredError) : 0;
    }

    internal override double CalculateR2()
    {
        double residualSumSquares = 0, totalSumSquares = 0;
        for (int i = 0; i < _sampleSize; i++)
        {
            residualSumSquares += Math.Pow(_oosActualValues[i] - _oosPredictions[i], 2);
            totalSumSquares += Math.Pow(_oosActualValues[i] - _oosActualValuesAvg, 2);
        }

        // We are saving these values for later reuse
        _residualSumOfSquares = residualSumSquares;
        _totalSumOfSquares = totalSumSquares;

        return _totalSumOfSquares != 0 ? 1 - (_residualSumOfSquares / _totalSumOfSquares) : 0;
    }

    internal override double CalculateAdjustedR2(double r2)
    {
        return _sampleSize != 1 && DegreesOfFreedom != 1 ? 1 - (1 - Math.Pow(r2, 2)) * (_sampleSize - 1) / (DegreesOfFreedom - 1) : 0;
    }

    internal override double CalculateAverageStandardError()
    {
        return AverageStandardDeviation / Math.Sqrt(_sampleSize);
    }

    internal override double CalculatePredictionStandardError()
    {
        return PredictionsStandardDeviation / Math.Sqrt(_sampleSize);
    }

    private static double CalculateStandardDeviation(double avgSumSquares)
    {
        return avgSumSquares >= 0 ? Math.Sqrt(avgSumSquares) : 0;
    }

    internal override double CalculateAverageStandardDeviation()
    {
        var avgSumSquares = _totalSumOfSquares / _sampleSize;

        return CalculateStandardDeviation(avgSumSquares);
    }

    internal override double CalculatePredictionStandardDeviation()
    {
        var avgSumSquares = _residualSumOfSquares / _sampleSize;

        return CalculateStandardDeviation(avgSumSquares);
    }

    internal override int CalculateDegreesOfFreedom()
    {
        return _sampleSize - _paramsCount;
    }
}