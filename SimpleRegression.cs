namespace AiDotNet;

public class SimpleRegression : IRegression
{
    public double YIntercept { get; private set; }

    public double Slope { get; private set; }

    public double[] Predictions { get; private set; }

    public IMetrics Metrics { get; private set; }

    private int TrainingSize { get; }

    private int OutOfSampleSize { get; }

    private double[] TrainingInputs { get; }
    private double[] OutOfSampleInputs { get; }
    private double[] TrainingOutputs { get; }
    private double[] OutOfSampleOutputs { get; }

    public SimpleRegression(double[] inputs, double[] outputs, int trainingPctSize = 25, INormalization? normalization = null)
    {
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
            throw new ArgumentException("Inputs and outputs must be the same length");
        }

        if (inputs.Length < 2)
        {
            throw new ArgumentException("Inputs and outputs need to have at least 2 values");
        }

        if (trainingPctSize < 1 || trainingPctSize > 99)
        {
            throw new ArgumentException(nameof(trainingPctSize), $"{nameof(trainingPctSize)} must be between 1 and 99");
        }

        TrainingSize = (int)Math.Floor((double)inputs.Length * trainingPctSize / 100);
        OutOfSampleSize = inputs.Length - TrainingSize;
        TrainingInputs = inputs.Take(TrainingSize).ToArray();
        TrainingOutputs = outputs.Take(TrainingSize).ToArray();
        OutOfSampleInputs = inputs.Skip(TrainingSize).ToArray();
        OutOfSampleOutputs = outputs.Skip(TrainingSize).ToArray();

        // todo: handle normalization if normalization isn't null before we fit the data

        (YIntercept, Slope) = Fit(TrainingInputs, TrainingOutputs);
        Predictions = Transform(OutOfSampleInputs);
        Metrics = new Metrics(Predictions, OutOfSampleOutputs);
    }

    internal sealed override (double yIntercept, double slope) Fit(double[] inputs, double[] outputs)
    {
        double inputAvg = 0, outputAvg = 0;
        for (var i = 0; i < inputs.Length; i++)
        {
            inputAvg += inputs[i];
            outputAvg += outputs[i];
        }

        inputAvg /= inputs.Length;
        outputAvg /= outputs.Length;

        double variance = 0, covariance = 0;
        for (var i = 0; i < inputs.Length; i++)
        {
            var diff = inputs[i] - inputAvg;
            variance += diff * diff;
            covariance += diff * (outputs[i] - outputAvg);
        }

        var m = variance != 0 ? covariance / variance : 0;

        return (outputAvg - m * inputAvg, m);
    }

    internal sealed override double[] Transform(double[] inputs)
    {
        var predictions = new double[inputs.Length];

        for (var i = 0; i < inputs.Length; i++)
        {
            predictions[i] = YIntercept + Slope * inputs[i];
        }

        return predictions;
    }
}