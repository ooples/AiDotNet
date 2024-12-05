using AiDotNet.LinearAlgebra;

namespace AiDotNet.Regression;

public sealed class PolynomialRegression : IRegression<double, double>
{
    private double YIntercept { get; set; }
    private double[] Coefficients { get; set; } = Array.Empty<double>();
    private MultipleRegressionOptions RegressionOptions { get; }
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
    /// Performs polynomial regression on the provided inputs and outputs. A polynomial regression is a form of regression analysis in which the relationship
    /// between the input and the output is modeled as an nth degree polynomial in the input.
    /// This handles all of the steps needed to create a trained ai model including training, normalizing, splitting, and transforming the data.
    /// </summary>
    /// <param name="inputs">The raw inputs (predicted values) to compare against the output values</param>
    /// <param name="outputs">The raw outputs (actual values) to compare against the input values</param>
    /// <param name="order">The degree/order of the polynomial to use for the regression</param>
    /// <param name="regressionOptions">Different options to allow full customization of the regression process</param>
    /// <exception cref="ArgumentNullException">The input array and/or output array is null</exception>
    /// <exception cref="ArgumentException">The input array or output array is either not the same length or doesn't have enough data</exception>
    public PolynomialRegression(double[] inputs, double[] outputs, int order, MultipleRegressionOptions? regressionOptions = null)
    {
        // do simple checks on all inputs and outputs before we do any work
        ValidationHelper.CheckForNullItems(inputs, outputs);
        var inputSize = inputs.Length;
        ValidationHelper.CheckForInvalidInputSize(inputSize, outputs.Length);

        // setting up default regression options if necessary
        RegressionOptions = regressionOptions ?? new MultipleRegressionOptions();

        // Check for invalid order such as a negative amount
        ValidationHelper.CheckForInvalidOrder(order, inputs);
        Order = order;

        // Check the training sizes to determine if we have enough training data to fit the model
        var trainingPctSize = RegressionOptions.TrainingPctSize;
        ValidationHelper.CheckForInvalidTrainingPctSize(trainingPctSize);
        var trainingSize = (int)Math.Floor(inputSize * trainingPctSize / 100);
        ValidationHelper.CheckForInvalidTrainingSizes(trainingSize, inputSize - trainingSize, Math.Min(2, inputSize), trainingPctSize);

        // Perform the actual work necessary to create the prediction and metrics models
        var (trainingInputs, trainingOutputs, oosInputs, oosOutputs) =
            PrepareData(inputs, outputs, trainingSize, RegressionOptions.Normalization);
        var (cleanedInputs, cleanedOutputs) = 
            RegressionOptions.OutlierRemoval?.RemoveOutliers(trainingInputs, trainingOutputs) ?? (trainingInputs, trainingOutputs);
        Fit(cleanedInputs, cleanedOutputs);
        Predictions = Transform(oosInputs);
        Metrics = new Metrics(Predictions, oosOutputs, inputSize, RegressionOptions.OutlierRemoval?.Quartile);
    }

    internal override void Fit(double[] inputs, double[] outputs)
    {
        var operations = MatrixHelper.GetNumericOperations<double>();
        var inputMatrix = new Matrix<double>(inputs.Length, Order + 1, operations);
        for (int i = 0; i < inputs.Length; i++)
        {
            for (int j = 0; j <= Order; j++)
            {
                inputMatrix[i, j] = Math.Pow(inputs[i], j);
            }
        }

        var outputVector = new Vector<double>(outputs, operations);
        var coefficients = new Vector<double>(Order + 1, operations);
        var cramerArray = new double[Order + 1];

        if (RegressionOptions.UseIntercept)
        {
            var onesVector = Vector<double>.Ones(RegressionOptions.MatrixLayout == MatrixLayout.ColumnArrays ? outputs.Length : Order + 1, operations);
            inputMatrix = RegressionOptions.MatrixLayout == MatrixLayout.ColumnArrays
                ? inputMatrix.InsertColumn(0, onesVector)
                : inputMatrix.InsertRow(0, onesVector);
        }

        switch (RegressionOptions.MatrixDecomposition)
        {
            case MatrixDecomposition.Cramer:
                cramerArray = new CramerMatrix(inputs, outputs, Order).Coefficients;
                break;
            case MatrixDecomposition.Cholesky:
                coefficients = inputMatrix.Cholesky().Solve(outputVector);
                break;
            case MatrixDecomposition.Evd:
                coefficients = inputMatrix.Evd().Solve(outputVector);
                break;
            case MatrixDecomposition.GramSchmidt:
                coefficients = inputMatrix.GramSchmidt().Solve(outputVector);
                break;
            case MatrixDecomposition.Lu:
                coefficients = inputMatrix.LU().Solve(outputVector);
                break;
            case MatrixDecomposition.Qr:
                coefficients = inputMatrix.QR().Solve(outputVector);
                break;
            case MatrixDecomposition.Svd:
                coefficients = inputMatrix.Svd().Solve(outputVector);
                break;
            default:
                coefficients = inputMatrix.Solve(outputVector);
                break;
        }

        Coefficients = RegressionOptions.MatrixDecomposition == MatrixDecomposition.Cramer ? cramerArray : coefficients.ToArray();
        YIntercept = RegressionOptions.UseIntercept ? Coefficients[0] : 0;
    }

    internal override (double[] trainingInputs, double[] trainingOutputs, double[] oosInputs, double[] oosOutputs) 
        PrepareData(double[] inputs, double[] outputs, int trainingSize, INormalizer? normalizer)
    {
        return normalizer?.PrepareData(inputs, outputs, trainingSize) ?? NormalizationHelper.SplitData(inputs, outputs, trainingSize);
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