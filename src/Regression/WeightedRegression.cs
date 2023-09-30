namespace AiDotNet.Regression;

public sealed class WeightedRegression : IRegression<double[], double>
{
    private double YIntercept { get; set; }
    private double[] Coefficients { get; set; } = Array.Empty<double>();
    private MultipleRegressionOptions RegressionOptions { get; }
    private double[] Weights { get; }

    public double[] Predictions { get; private set; }
    public IMetrics Metrics { get; private set; }

    public WeightedRegression(double[][] inputs, double[] outputs, double[] weights, MultipleRegressionOptions? regressionOptions = null)
    {
        // do simple checks on all inputs and outputs before we do any work
        ValidationHelper.CheckForNullItems(inputs, outputs, weights);
        var inputSize = inputs[0].Length;
        ValidationHelper.CheckForInvalidInputSize(inputSize, outputs.Length);
        Weights = weights;

        // setting up default regression options if necessary
        RegressionOptions = regressionOptions ?? new MultipleRegressionOptions();

        // Check the training sizes to determine if we have enough training data to fit the model
        var trainingPctSize = RegressionOptions.TrainingPctSize;
        ValidationHelper.CheckForInvalidTrainingPctSize(trainingPctSize);
        var trainingSize = (int)Math.Floor(inputSize * trainingPctSize / 100);
        ValidationHelper.CheckForInvalidTrainingSizes(trainingSize, inputSize - trainingSize, Math.Max(2, inputs.Length), trainingPctSize);

        // Perform the actual work necessary to create the prediction and metrics models
        var (trainingInputs, trainingOutputs, oosInputs, oosOutputs) =
            PrepareData(inputs, outputs, trainingSize, RegressionOptions.Normalization);
        Fit(trainingInputs, trainingOutputs);
        Predictions = Transform(oosInputs);
        Metrics = new Metrics(Predictions, oosOutputs, inputs.Length);
    }

    internal override (double[][] trainingInputs, double[] trainingOutputs, double[][] oosInputs, double[] oosOutputs) PrepareData(
        double[][] inputs, double[] outputs, int trainingSize, INormalization? normalization)
    {
        return normalization?.PrepareData(inputs, outputs, trainingSize) ?? NormalizationHelper.SplitData(inputs, outputs, trainingSize);
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
        var weights = m.Diagonal(Weights);

        if (RegressionOptions.UseIntercept)
        {
            inputMatrix = RegressionOptions.MatrixLayout == MatrixLayout.ColumnArrays ?
                inputMatrix.InsertColumn(0, CreateVector.Dense(outputs.Length, Vector<double>.One)) :
                inputMatrix.InsertRow(0, CreateVector.Dense(outputs.Length, Vector<double>.One));
        }

        var result = RegressionOptions.MatrixDecomposition switch
        {
            MatrixDecomposition.Cholesky => inputMatrix.TransposeThisAndMultiply(weights * inputMatrix).Cholesky()
                                .Solve(inputMatrix.TransposeThisAndMultiply(weights * outputVector)),
            MatrixDecomposition.Evd => inputMatrix.TransposeThisAndMultiply(weights * inputMatrix).Evd()
                                .Solve(inputMatrix.TransposeThisAndMultiply(weights * outputVector)),
            MatrixDecomposition.GramSchmidt => inputMatrix.TransposeThisAndMultiply(weights * inputMatrix).GramSchmidt()
                                .Solve(inputMatrix.TransposeThisAndMultiply(weights * outputVector)),
            MatrixDecomposition.Lu => inputMatrix.TransposeThisAndMultiply(weights * inputMatrix).LU()
                                .Solve(inputMatrix.TransposeThisAndMultiply(weights * outputVector)),
            MatrixDecomposition.Qr => inputMatrix.TransposeThisAndMultiply(weights * inputMatrix).QR()
                                .Solve(inputMatrix.TransposeThisAndMultiply(weights * outputVector)),
            MatrixDecomposition.Svd => inputMatrix.TransposeThisAndMultiply(weights * inputMatrix).Svd()
                                .Solve(inputMatrix.TransposeThisAndMultiply(weights * outputVector)),
            _ => inputMatrix.TransposeThisAndMultiply(weights * inputMatrix).Cholesky()
                                .Solve(inputMatrix.TransposeThisAndMultiply(weights * outputVector)),
        };

        Coefficients = result.ToArray();
        YIntercept = 0;
    }

    internal override double[] Transform(double[][] inputs)
    {
        var predictions = new double[inputs[0].Length];

        for (var i = 0; i < inputs.Length; i++)
        {
            for (var j = 0; j < inputs[j].Length; j++)
            {
                predictions[j] += YIntercept + Coefficients[i] * inputs[i][j];
            }
        }

        return predictions;
    }
}