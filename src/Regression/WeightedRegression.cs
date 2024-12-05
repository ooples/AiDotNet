namespace AiDotNet.Regression;

public sealed class WeightedRegression : IRegression<double, double>
{
    private double YIntercept { get; set; }
    private double[] Coefficients { get; set; } = Array.Empty<double>();
    private MultipleRegressionOptions RegressionOptions { get; }
    private double[] Weights { get; }
    private int Order { get; }

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
    /// Performs a weighted regression on the provided inputs and outputs. A weighted regression multiplies each input by a weight to give it more or less importance.
    /// This handles all of the steps needed to create a trained ai model including training, normalizing, splitting, and transforming the data.
    /// </summary>
    /// <param name="inputs">The raw inputs (predicted values) to compare against the output values</param>
    /// <param name="outputs">The raw outputs (actual values) to compare against the input values</param>
    /// <param name="weights">The raw weights to apply to each input</param>
    /// <param name="regressionOptions">Different options to allow full customization of the regression process</param>
    /// <exception cref="ArgumentNullException">The input array and/or output array is null</exception>
    /// <exception cref="ArgumentException">The input array or output array is either not the same length or doesn't have enough data</exception>
    public WeightedRegression(double[] inputs, double[] outputs, double[] weights, int order, MultipleRegressionOptions? regressionOptions = null)
    {
        // do simple checks on all inputs and outputs before we do any work
        ValidationHelper.CheckForNullItems(inputs, outputs);
        var inputSize = inputs.Length;
        ValidationHelper.CheckForInvalidInputSize(inputSize, outputs.Length);
        ValidationHelper.CheckForInvalidWeights(weights);
        Weights = weights;

        // setting up default regression options if necessary
        RegressionOptions = regressionOptions ?? new MultipleRegressionOptions();

        // Check for invalid order such as a negative amount
        ValidationHelper.CheckForInvalidOrder(order, inputs);
        Order = order;

        // Check the training sizes to determine if we have enough training data to fit the model
        var trainingPctSize = RegressionOptions.TrainingPctSize;
        ValidationHelper.CheckForInvalidTrainingPctSize(trainingPctSize);
        var trainingSize = (int)Math.Floor(inputSize * trainingPctSize / 100);
        ValidationHelper.CheckForInvalidTrainingSizes(trainingSize, inputSize - trainingSize, Math.Min(2, inputs.Length), trainingPctSize);

        // Perform the actual work necessary to create the prediction and metrics models
        var (trainingInputs, trainingOutputs, oosInputs, oosOutputs) =
            PrepareData(inputs, outputs, trainingSize, RegressionOptions.Normalization);
        var (cleanedInputs, cleanedOutputs) = 


            RegressionOptions.OutlierRemoval?.RemoveOutliers(trainingInputs, trainingOutputs) ?? (trainingInputs, trainingOutputs);
        Fit(cleanedInputs, cleanedOutputs);
        Predictions = Transform(oosInputs);
        Metrics = new Metrics(Predictions, oosOutputs, inputs.Length, RegressionOptions.OutlierRemoval?.Quartile);
    }

    internal override (double[] trainingInputs, double[] trainingOutputs, double[] oosInputs, double[] oosOutputs) PrepareData(
        double[] inputs, double[] outputs, int trainingSize, INormalizer? normalizer)
    {
        return normalizer?.PrepareData(inputs, outputs, trainingSize) ?? NormalizationHelper.SplitData(inputs, outputs, trainingSize);
    }

    internal override void Fit(double[] inputs, double[] outputs)
    {
        var operations = MathHelper.GetNumericOperations<double>();
        var inputMatrix = new Matrix<double>(inputs.Length, Order + 1, operations);
        for (int i = 0; i < inputs.Length; i++)
        {
            for (int j = 0; j <= Order; j++)
            {
                inputMatrix[i, j] = Math.Pow(inputs[i], j);
            }
        }

        var outputVector = new Vector<double>(outputs, operations);
        var weights = new Matrix<double>(inputs.Length, inputs.Length, operations);
        for (int i = 0; i < inputs.Length; i++)
        {
            weights[i, i] = Weights[i];
        }

        if (RegressionOptions.UseIntercept)
        {
            var onesVector = new Vector<double>(outputs.Length, operations);
            for (int i = 0; i < onesVector.Length; i++)
            {
                onesVector[i] = 1.0;
            }

            inputMatrix = RegressionOptions.MatrixLayout == MatrixLayout.ColumnArrays ?
                inputMatrix.InsertColumn(0, onesVector) :
                inputMatrix.InsertRow(0, onesVector);
        }

        var transposedInputMatrix = inputMatrix.Transpose();
        var weightedInputMatrix = weights * inputMatrix;
        var leftSide = transposedInputMatrix * weightedInputMatrix;
        var rightSide = transposedInputMatrix * (weights * outputVector);

        Vector<double> result;

        switch (RegressionOptions.MatrixDecomposition)
        {
            case MatrixDecomposition.Cholesky:
                var choleskyDecomp = new CholeskyDecomposition(leftSide);
                result = choleskyDecomp.Solve(rightSide);
                break;
            case MatrixDecomposition.Lu:
                var luDecomp = new LuDecomposition(leftSide);
                result = luDecomp.Solve(rightSide);
                break;
            // Implement other decomposition methods as needed
            default:
                // Use a default method, e.g., Gaussian elimination
                result = GaussianElimination(leftSide, rightSide);
                break;
        }

        Coefficients = result.ToArray();
        YIntercept = RegressionOptions.UseIntercept ? Coefficients[0] : 0;
    }

    internal override double[] Transform(double[] inputs)
    {
        var predictions = new double[inputs.Length];

        for (var i = 0; i < inputs.Length; i++)
        {
            for (var j = 0; j < Order + 1; j++)
            {
                predictions[j] += YIntercept + Coefficients[j] * inputs[i];
            }
        }

        return predictions;
    }
}