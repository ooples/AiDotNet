using AiDotNet.Models;
using AiDotNet.Statistics;

namespace AiDotNet.Regression;

public sealed class SimpleRegression : IRegression<double, double>
{
    private double YIntercept { get; set; }
    private double Slope { get; set; }

    public double[] Predictions { get; private set; }
    public IMetrics Metrics { get; private set; }

    /// <summary>
    /// Performs a simple linear regression on the provided inputs and outputs.
    /// This handles all of the steps needed to create a trained ai model including training, normalizing, splitting, and transforming the data.
    /// </summary>
    /// <param name="inputs">The raw inputs (predicted values) to compare against the output values</param>
    /// <param name="outputs">The raw outputs (actual values) to compare against the input values</param>
    /// <param name="regressionOptions">Different options to allow full customization of the regression process</param>
    /// <exception cref="ArgumentNullException">The input array and/or output array is null</exception>
    /// <exception cref="ArgumentException">The input array or output array is either not the same length or doesn't have enough data</exception>
    public SimpleRegression(double[] inputs, double[] outputs, SimpleRegressionOptions? regressionOptions = null)
    {
        regressionOptions ??= new SimpleRegressionOptions();

        if (inputs == null)
        {
            throw new ArgumentNullException(nameof(inputs), "Inputs can't be null");
        }

        if (outputs == null)
        {
            throw new ArgumentNullException(nameof(outputs), "Outputs can't be null");
        }

        if (inputs.Length != outputs.Length)
        {
            throw new ArgumentException("Inputs and outputs must have the same length");
        }

        if (inputs.Length < 2)
        {
            throw new ArgumentException("Inputs and outputs must have at least 2 values each");
        }

        var trainingPctSize = regressionOptions.TrainingPctSize;
        if (trainingPctSize <= 0 || trainingPctSize >= 100)
        {
            throw new ArgumentException($"{nameof(trainingPctSize)} must be greater than 0 and less than 100", nameof(trainingPctSize));
        }

        // Set the training sizes to determine if we have enough training data to fit the model
        var trainingSize = (int)Math.Floor(inputs.Length * trainingPctSize / 100);
        var outOfSampleSize = inputs.Length - trainingSize;

        if (trainingSize < 2)
        {
            throw new ArgumentException($"Training data must contain at least 2 values. " +
                                        $"You either need to increase your {nameof(trainingPctSize)} or increase the amount of inputs and outputs data");
        }

        if (outOfSampleSize < 2)
        {
            throw new ArgumentException($"Out of sample data must contain at least 2 values. " +
                                        $"You either need to decrease your {nameof(trainingPctSize)} or increase the amount of inputs and outputs data");
        }

        var (trainingInputs, trainingOutputs, oosInputs, oosOutputs) = 
            PrepareData(inputs, outputs, trainingSize, regressionOptions.Normalization);
        Fit(trainingInputs, trainingOutputs);
        Predictions = Transform(oosInputs);
        Metrics = new Metrics(Predictions, oosOutputs, inputs.Rank);
    }

    internal override void Fit(double[] x, double[] y)
    {
        var n = x.Length;
        double sumX = 0, sumY = 0, sumXy = 0, sumXSq = 0;
        for (var i = 0; i < n; i++)
        {
            sumX += x[i];
            sumY += y[i];
            sumXy += x[i] * y[i];
            sumXSq += x[i] * x[i];
        }

        Slope = (n * sumXy - sumX * sumY) / (n * sumXSq - sumX * sumX);
        YIntercept = (sumY - Slope * sumX) / n;
    }

    internal override double[] Transform(double[] inputs)
    {
        var predictions = new double[inputs.Length];

        for (var i = 0; i < inputs.Length; i++)
        {
            predictions[i] = YIntercept + Slope * inputs[i];
        }

        return predictions;
    }

    internal override (double[], double[], double[], double[]) PrepareData(double[] inputs, double[] outputs, int trainingSize, INormalization? normalization)
    {
        return normalization?.PrepareData(inputs, outputs, trainingSize) ?? NormalizationHelper.SplitData(inputs, outputs, trainingSize);
    }
}