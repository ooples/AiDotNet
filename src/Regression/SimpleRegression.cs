namespace AiDotNet.Regression;

public sealed class SimpleRegression : IRegression<double, double>
{
    private double YIntercept { get; set; }
    private double Slope { get; set; }
    private SimpleRegressionOptions RegressionOptions { get; }

    /// <summary>
    /// Predictions created from the out of sample (oos) data only.
    /// </summary>
    public double[] Predictions { get; private set; }

    /// <summary>
    /// Metrics data to help evaluate the performance of a model by comparing the predicted values to the actual values.
    /// Predicted values are taken from the out of sample (oos) data only.
    /// </summary>
    public Metrics Metrics { get; private set; }

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
        // do simple checks on all inputs and outputs before we do any work
        ValidationHelper.CheckForNullItems(inputs, outputs);
        var inputSize = inputs.Length;
        ValidationHelper.CheckForInvalidInputSize(inputSize, outputs.Length);

        // setting up default regression options if necessary
        RegressionOptions = regressionOptions ?? new SimpleRegressionOptions();

        // Check the training sizes to determine if we have enough training data to fit the model
        var trainingPctSize = RegressionOptions.TrainingPctSize;
        ValidationHelper.CheckForInvalidTrainingPctSize(trainingPctSize);
        var trainingSize = (int)Math.Floor(inputSize * trainingPctSize / 100);
        ValidationHelper.CheckForInvalidTrainingSizes(trainingSize, inputSize - trainingSize, Math.Min(2, inputs.Length), trainingPctSize);

        // Perform the actual work necessary to create the prediction and metrics models
        var (trainingInputs, trainingOutputs, oosInputs, oosOutputs) =
            PrepareData(inputs, outputs, trainingSize, RegressionOptions.Normalization);
        Fit(trainingInputs, trainingOutputs);
        Predictions = Transform(oosInputs);
        Metrics = new Metrics(Predictions, oosOutputs, inputs.Length);
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