namespace AiDotNet.Helpers;

public static class ValidationHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public static void ValidateInputData(Matrix<T> x, Vector<T> y)
    {
        ValidateMatrixVectorPair(x, y, "Input");
    }

    public static void ValidateInputData(OptimizationInputData<T> inputData)
    {
        if (inputData == null)
            throw new ArgumentNullException(nameof(inputData), "Optimization input data cannot be null.");

        ValidateMatrixVectorPair(inputData.XTrain, inputData.YTrain, "Training");
        ValidateMatrixVectorPair(inputData.XVal, inputData.YVal, "Validation");
        ValidateMatrixVectorPair(inputData.XTest, inputData.YTest, "Test");

        // Ensure all matrices have the same number of columns
        if (inputData.XTrain.Columns != inputData.XVal.Columns || inputData.XTrain.Columns != inputData.XTest.Columns)
            throw new ArgumentException("All input matrices must have the same number of columns.");
    }

    public static void ValidatePoissonData(Vector<T> y)
    {
        for (int i = 0; i < y.Length; i++)
        {
            if (NumOps.LessThan(y[i], NumOps.Zero) || !MathHelper.IsInteger(y[i], NumOps))
            {
                throw new ArgumentException("Poisson regression requires non-negative integer response values.");
            }
        }
    }

    private static void ValidateMatrixVectorPair(Matrix<T> x, Vector<T> y, string datasetName)
    {
        if (x == null)
            throw new ArgumentNullException(nameof(x), $"{datasetName} matrix cannot be null.");

        if (y == null)
            throw new ArgumentNullException(nameof(y), $"{datasetName} target vector cannot be null.");

        if (x.Rows != y.Length)
            throw new ArgumentException($"Number of rows in {datasetName.ToLower()} matrix must match the length of the {datasetName.ToLower()} target vector.");

        if (x.Rows == 0 || x.Columns == 0)
            throw new ArgumentException($"{datasetName} matrix cannot be empty.");
    }
}