namespace AiDotNet.Regression;

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
        var (cleanedInputs, cleanedOutputs) = RegressionOptions.OutlierRemoval?.RemoveOutliers(inputs, outputs) ?? (inputs, outputs);
        var (normalizedInputs, normalizedOutputs, oosInputs, oosOutputs) =
            PrepareData(cleanedInputs, cleanedOutputs, trainingSize, RegressionOptions.Normalization);
        Fit(normalizedInputs, normalizedOutputs);
        Predictions = Transform(oosInputs);
        Metrics = new Metrics(Predictions, oosOutputs, inputSize, RegressionOptions.OutlierRemoval?.Quartile);
    }

    internal override void Fit(double[][] inputs, double[][] outputs)
    {
        var m = Matrix<double>.Build;
        var inputMatrix = RegressionOptions.MatrixLayout switch
        {
            MatrixLayout.ColumnArrays => m.DenseOfColumnArrays(inputs),
            MatrixLayout.RowArrays => m.DenseOfRowArrays(inputs),
            _ => m.DenseOfColumnArrays(inputs)
        };
        var outputMatrix = RegressionOptions.MatrixLayout switch
        {
            MatrixLayout.ColumnArrays => m.DenseOfColumnArrays(outputs),
            MatrixLayout.RowArrays => m.DenseOfRowArrays(outputs),
            _ => m.DenseOfColumnArrays(outputs)
        };

        if (RegressionOptions.UseIntercept)
        {
            inputMatrix = RegressionOptions.MatrixLayout == MatrixLayout.ColumnArrays ?
                inputMatrix.InsertColumn(0, CreateVector.Dense(outputs.Length, Vector<double>.One)) :
                inputMatrix.InsertRow(0, CreateVector.Dense(outputs.Length, Vector<double>.One));
        }

        var result = m.DenseOfMatrix(inputMatrix);
        switch (RegressionOptions.MatrixDecomposition)
        {
            case MatrixDecomposition.Cholesky:
                inputMatrix.Cholesky().Solve(outputMatrix, result);
                break;
            case MatrixDecomposition.Evd:
                inputMatrix.Evd().Solve(outputMatrix, result);
                break;
            case MatrixDecomposition.GramSchmidt:
                inputMatrix.GramSchmidt().Solve(outputMatrix, result);
                break;
            case MatrixDecomposition.Lu:
                inputMatrix.LU().Solve(outputMatrix, result);
                break;
            case MatrixDecomposition.Qr:
                inputMatrix.QR().Solve(outputMatrix, result);
                break;
            case MatrixDecomposition.Svd:
                inputMatrix.Svd().Solve(outputMatrix, result);
                break;
            default:
                inputMatrix.Solve(outputMatrix, result);
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