namespace AiDotNet.Optimizers;

public sealed class MultivariateRegression : IRegression<double[], double[]>
{
    private double YIntercept { get; set; }
    private double[][] Coefficients { get; set; } = Array.Empty<double[]>();
    private MultipleRegressionOptions RegressionOptions { get; }

    /// <summary>
    /// Predictions created from the out of sample (oos) data only.
    /// </summary>
    public double[][] Predictions { get; private set; }

    /// <summary>
    /// Metrics data to help evaluate the performance of a model by comparing the predicted values to the actual values.
    /// Predicted values are taken from the out of sample (oos) data only.
    /// </summary>
    public Metrics Metrics { get; private set; }

    /// <summary>
    /// Performs multivariate regression on the provided inputs and outputs.
    /// Multivariate regression is the same as multiple regression except that the output is a multi-dimensional array of values instead of a one-dimensional array.
    /// This handles all of the steps needed to create a trained ai model including training, normalizing, splitting, and transforming the data.
    /// </summary>
    /// <param name="inputs">The raw inputs (predicted values) to compare against the output values</param>
    /// <param name="outputs">The raw outputs (actual values) to compare against the input values</param>
    /// <param name="regressionOptions">Different options to allow full customization of the regression process</param>
    /// <exception cref="ArgumentNullException">The input array and/or output array is null</exception>
    /// <exception cref="ArgumentException">The input array or output array is either not the same length or doesn't have enough data</exception>
    public MultivariateRegression(double[][] inputs, double[][] outputs, MultipleRegressionOptions? regressionOptions = null)
    {
        // do simple checks on all inputs and outputs before we do any work
        ValidationHelper.CheckForNullItems(inputs, outputs);
        var inputSize = inputs[0].Length;
        ValidationHelper.CheckForInvalidInputSize(inputSize, outputs[0].Length);

        // setting up default regression options if necessary
        RegressionOptions = regressionOptions ?? new MultipleRegressionOptions();

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

    internal override void Fit(double[][] inputs, double[][] outputs)
    {
        var inputMatrix = RegressionOptions.MatrixLayout switch
        {
            MatrixLayout.ColumnArrays => new Matrix<double>(inputs),
            MatrixLayout.RowArrays => new Matrix<double>(inputs).Transpose(),
            _ => new Matrix<double>(inputs)
        };
        var outputMatrix = RegressionOptions.MatrixLayout switch
        {
            MatrixLayout.ColumnArrays => new Matrix<double>(outputs),
            MatrixLayout.RowArrays => new Matrix<double>(outputs).Transpose(),
            _ => new Matrix<double>(outputs)
        };

        if (RegressionOptions.UseIntercept)
        {
            var onesVector = VectorHelper.CreateVector<double>(outputs.Length);
            for (int i = 0; i < onesVector.Length; i++)
            {
                onesVector[i] = 1.0;
            }

            inputMatrix = RegressionOptions.MatrixLayout == MatrixLayout.ColumnArrays ?
                inputMatrix.InsertColumn(0, onesVector) :
                inputMatrix.InsertRow(0, onesVector);
        }

        var result = new Matrix<double>(inputMatrix.Rows, outputMatrix.Columns);
        switch (RegressionOptions.MatrixDecomposition)
        {
            case MatrixDecomposition.Cholesky:
                result = inputMatrix.Cholesky().Solve(outputMatrix);
                break;
            case MatrixDecomposition.Evd:
                result = inputMatrix.Evd().Solve(outputMatrix);
                break;
            case MatrixDecomposition.GramSchmidt:
                result = inputMatrix.GramSchmidt().Solve(outputMatrix);
                break;
            case MatrixDecomposition.Lu:
                result = inputMatrix.Lu().Solve(outputMatrix);
                break;
            case MatrixDecomposition.Qr:
                result = inputMatrix.Qr().Solve(outputMatrix);
                break;
            case MatrixDecomposition.Svd:
                result = inputMatrix.Svd().Solve(outputMatrix);
                break;
            default:
                result = inputMatrix.Solve(outputMatrix);
                break;
        }

        Coefficients = RegressionOptions.MatrixLayout == MatrixLayout.ColumnArrays ? result.ToColumnArrays() : result.ToRowArrays();
        YIntercept = 0;
    }


    internal override (double[][] trainingInputs, double[][] trainingOutputs, double[][] oosInputs, double[][] oosOutputs)
        PrepareData(double[][] inputs, double[][] outputs, int trainingSize, INormalization? normalization)
    {
        return normalization?.PrepareData(inputs, outputs, trainingSize) ?? NormalizationHelper.SplitData(inputs, outputs, trainingSize);
    }

    internal override double[][] Transform(double[][] inputs)
    {
        var predictions = new double[inputs.Length][];
        predictions[0] = new double[inputs[0].Length];
        predictions[1] = new double[inputs[1].Length];

        for (var i = 0; i < inputs.Length; i++)
        {
            for (var j = 0; j < inputs[i].Length; j++)
            {
                predictions[i][j] = Coefficients[i][1] + Coefficients[i][0] * inputs[i][j];
            }
        }

        return predictions;
    }
}