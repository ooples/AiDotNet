namespace AiDotNet.Regression;

public sealed class MultipleRegression : IRegression<double[], double>
{
    private double YIntercept { get; set; }
    private double[] Coefficients { get; set; } = Array.Empty<double>();
    private MultipleRegressionOptions RegressionOptions { get; }

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
    /// Performs multiple regression on the provided inputs and outputs.
    /// This handles all of the steps needed to create a trained ai model including training, normalizing, splitting, and transforming the data.
    /// </summary>
    /// <param name="inputs">The raw inputs (predicted values) to compare against the output values</param>
    /// <param name="outputs">The raw outputs (actual values) to compare against the input values</param>
    /// <param name="regressionOptions">Different options to allow full customization of the regression process</param>
    /// <exception cref="ArgumentNullException">The input array and/or output array is null</exception>
    /// <exception cref="ArgumentException">The input array or output array is either not the same length or doesn't have enough data</exception>
    public MultipleRegression(double[][] inputs, double[] outputs, MultipleRegressionOptions? regressionOptions = null)
    {
        // do simple checks on all inputs and outputs before we do any work
        ValidationHelper.CheckForNullItems(inputs, outputs);
        var inputSize = inputs[0].Length;
        ValidationHelper.CheckForInvalidInputSize(inputSize, outputs.Length);

        // setting up default regression options if necessary
        RegressionOptions = regressionOptions ?? new MultipleRegressionOptions();

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

    internal override void Fit(double[][] inputs, double[] outputs)
    {
        var m = Matrix<double>.Build;
        var inputMatrix = RegressionOptions.MatrixLayout switch
        {
            MatrixLayout.ColumnArrays => m.DenseOfColumnArrays(inputs),
            MatrixLayout.RowArrays => m.DenseOfRowArrays(inputs),
            _ => m.DenseOfColumnArrays(inputs)
        };
        var outputVector = CreateVector.Dense(outputs);
        var result = CreateVector.Dense<double>(inputs.Length + (RegressionOptions.UseIntercept ? 1 : 0));

        if (RegressionOptions.UseIntercept)
        {
            inputMatrix = RegressionOptions.MatrixLayout == MatrixLayout.ColumnArrays ? 
                inputMatrix.InsertColumn(0, CreateVector.Dense(outputs.Length, Vector<double>.One)) :
                inputMatrix.InsertRow(0, CreateVector.Dense(outputs.Length, Vector<double>.One));
        }

        switch (RegressionOptions.MatrixDecomposition)
        {
            case MatrixDecomposition.Cholesky:
                inputMatrix.Cholesky().Solve(outputVector, result);
                break;
            case MatrixDecomposition.Evd:
                inputMatrix.Evd().Solve(outputVector, result);
                break;
            case MatrixDecomposition.GramSchmidt:
                inputMatrix.GramSchmidt().Solve(outputVector, result);
                break;
            case MatrixDecomposition.Lu:
                inputMatrix.LU().Solve(outputVector, result);
                break;
            case MatrixDecomposition.Qr:
                inputMatrix.QR().Solve(outputVector, result);
                break;
            case MatrixDecomposition.Svd:
                inputMatrix.Svd().Solve(outputVector, result);
                break;
            default:
                inputMatrix.Solve(outputVector, result);
                break;
        }

        Coefficients = result.ToArray();
        YIntercept = 0;
    }

    internal override (double[][] trainingInputs, double[] trainingOutputs, double[][] oosInputs, double[] oosOutputs) 
        PrepareData(double[][] inputs, double[] outputs, int trainingSize, INormalization? normalization)
    {
        return normalization?.PrepareData(inputs, outputs, trainingSize) ?? NormalizationHelper.SplitData(inputs, outputs, trainingSize);
    }

    internal override double[] Transform(double[][] inputs)
    {
        var predictions = new double[inputs[0].Length];

        for (var i = 0; i < inputs.Length; i++)
        {
            for (var j = 0; j < inputs[i].Length; j++)
            {
                predictions[j] += Coefficients[i] * inputs[i][j];
            }
        }

        return predictions;
    }
}