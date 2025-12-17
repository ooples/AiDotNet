namespace AiDotNet.Helpers;

/// <summary>
/// Provides validation methods for AI model inputs and parameters.
/// </summary>
/// <typeparam name="T">The numeric type used in calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <b>For Beginners:</b> This helper class ensures that the data you provide to AI models is valid and properly formatted.
/// It can handle both traditional matrix/vector inputs (for regression-like models) and tensor inputs (for neural networks).
/// Think of it as a quality control checkpoint that prevents errors before they happen by checking that your
/// data meets all the requirements needed for successful model training and prediction.
/// </remarks>
public static class ValidationHelper<T>
{
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Validates that input data is properly formatted for model training.
    /// </summary>
    /// <typeparam name="TInput">The type of the input data (e.g., Matrix&lt;T&gt; or Tensor&lt;T&gt;).</typeparam>
    /// <typeparam name="TOutput">The type of the output data (e.g., Vector&lt;T&gt; or Tensor&lt;T&gt;).</typeparam>
    /// <param name="x">The input data.</param>
    /// <param name="y">The target data.</param>
    /// <remarks>
    /// <b>For Beginners:</b> This method checks that your input data (x) and output data (y) are compatible.
    /// It can handle both traditional matrix/vector pairs (for regression-like models) and tensor pairs (for neural networks).
    /// The method ensures they have matching dimensions and are not null or empty.
    /// </remarks>
    public static void ValidateInputData<TInput, TOutput>(TInput x, TOutput y)
    {
        ValidateDataPair(x, y, "Input");
    }

    /// <summary>
    /// Gets information about the calling method.
    /// </summary>
    /// <param name="skipFrames">Number of frames to skip in the stack trace.</param>
    /// <returns>A tuple containing the component name and operation name.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method identifies which part of the code called a validation function.
    /// It's like caller ID for functions, helping to provide more specific error messages.
    /// You typically won't need to call this directly in your code.
    /// </remarks>
    public static (string component, string operation) GetCallerInfo(int skipFrames = 2)
    {
        try
        {
            // Skip the specified number of frames to get to the actual client code
            var stackTrace = new StackTrace(skipFrames, false);
            var frame = stackTrace.GetFrame(0);

            if (frame != null)
            {
                var method = frame.GetMethod();
                if (method != null)
                {
                    string operation = method.Name;
                    string component = method.DeclaringType?.Name ?? "Unknown";

                    return (component, operation);
                }
            }
        }
        catch (Exception)
        {
            // Fallback if stack trace inspection fails
        }

        // Default values if we can't determine the caller
        return ("Unknown", "Validation");
    }

    /// <summary>
    /// Resolves component and operation names, using caller info if either is empty.
    /// </summary>
    /// <param name="component">The component name, or empty to use caller info.</param>
    /// <param name="operation">The operation name, or empty to use caller info.</param>
    /// <param name="skipFrames">Number of frames to skip in the stack trace.</param>
    /// <returns>A tuple containing the resolved component and operation names.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method helps create more informative error messages by identifying
    /// which part of the library is performing an operation. You typically won't need to call
    /// this directly in your code.
    /// </remarks>
    public static (string component, string operation) ResolveCallerInfo(string component = "", string operation = "", int skipFrames = 3)
    {
        // Only get caller info if needed
        if (string.IsNullOrEmpty(component) || string.IsNullOrEmpty(operation))
        {
            var callerInfo = GetCallerInfo(skipFrames);

            // Only use caller info for empty parameters
            if (string.IsNullOrEmpty(component))
                component = callerInfo.component;

            if (string.IsNullOrEmpty(operation))
                operation = callerInfo.operation;
        }

        return (component, operation);
    }

    /// <summary>
    /// Validates that optimization input data is properly formatted for model training and evaluation.
    /// </summary>
    /// <typeparam name="TInput">The type of the input data (e.g., Matrix&lt;T&gt; or Tensor&lt;T&gt;).</typeparam>
    /// <typeparam name="TOutput">The type of the output data (e.g., Vector&lt;T&gt; or Tensor&lt;T&gt;).</typeparam>
    /// <param name="inputData">The optimization input data containing training, validation, and test datasets.</param>
    /// <remarks>
    /// <b>For Beginners:</b> When training AI models, we typically split our data into three sets:
    /// 1. Training data - used to teach the model patterns (like studying for a test)
    /// 2. Validation data - used to tune the model (like practice tests)
    /// 3. Test data - used to evaluate the final model (like the final exam)
    /// 
    /// This method checks that all three datasets are properly formatted and compatible with each other.
    /// It can handle both matrix/vector pairs and tensor pairs.
    /// </remarks>
    public static void ValidateInputData<TInput, TOutput>(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        if (inputData == null)
            throw new ArgumentNullException(nameof(inputData), "Optimization input data cannot be null.");

        ValidateDataPair(inputData.XTrain, inputData.YTrain, "Training");
        ValidateDataPair(inputData.XValidation, inputData.YValidation, "Validation");
        ValidateDataPair(inputData.XTest, inputData.YTest, "Test");

        // Ensure all inputs have the same shape
        EnsureConsistentInputShape<TInput, TOutput>(inputData.XTrain, inputData.XValidation, inputData.XTest);
    }

    /// <summary>
    /// Validates that data is appropriate for Poisson regression.
    /// </summary>
    /// <param name="y">The target vector containing output values to predict.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Poisson regression is a special type of model used when predicting counts
    /// (like number of customers, number of events, etc.). This method checks that your target values
    /// are non-negative integers (0, 1, 2, etc.), which is required for Poisson regression to work correctly.
    /// 
    /// For example, if you're predicting "number of website visitors per day", each value must be
    /// a whole number (you can't have 3.5 visitors) and can't be negative (you can't have -2 visitors).
    /// </remarks>
    public static void ValidatePoissonData(Vector<T> y)
    {
        for (int i = 0; i < y.Length; i++)
        {
            if (_numOps.LessThan(y[i], _numOps.Zero) || !MathHelper.IsInteger(y[i]))
            {
                throw new ArgumentException("Poisson regression requires non-negative integer response values.");
            }
        }
    }

    private static void ValidateDataPair<TInput, TOutput>(TInput x, TOutput y, string datasetName)
    {
        if (x is Matrix<T> xMatrix && y is Vector<T> yVector)
        {
            ValidateMatrixVectorPair(xMatrix, yVector, datasetName);
        }
        else if (x is Tensor<T> xTensor && y is Tensor<T> yTensor)
        {
            ValidateTensorPair(xTensor, yTensor, datasetName);
        }
        else
        {
            throw new ArgumentException($"Invalid input types for {datasetName} dataset. Expected Matrix<T> and Vector<T>, or Tensor<T> and Tensor<T>.");
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

    private static void ValidateTensorPair(Tensor<T> x, Tensor<T> y, string datasetName)
    {
        if (x == null)
            throw new ArgumentNullException(nameof(x), $"{datasetName} input tensor cannot be null.");

        if (y == null)
            throw new ArgumentNullException(nameof(y), $"{datasetName} target tensor cannot be null.");

        if (x.Shape[0] != y.Shape[0])
            throw new ArgumentException($"First dimension of {datasetName.ToLower()} input tensor must match the first dimension of the {datasetName.ToLower()} target tensor.");

        if (x.Shape.Any(dim => dim == 0) || y.Shape.Any(dim => dim == 0))
            throw new ArgumentException($"{datasetName} tensors cannot have zero-sized dimensions.");
    }

    private static void EnsureConsistentInputShape<TInput, TOutput>(TInput xTrain, TInput xValidation, TInput xTest)
    {
        if (xTrain is Matrix<T> xTrainMatrix && xValidation is Matrix<T> xValMatrix && xTest is Matrix<T> xTestMatrix)
        {
            if (xTrainMatrix.Columns != xValMatrix.Columns || xTrainMatrix.Columns != xTestMatrix.Columns)
                throw new ArgumentException("All input matrices must have the same number of columns.");
        }
        else if (xTrain is Tensor<T> xTrainTensor && xValidation is Tensor<T> xValTensor && xTest is Tensor<T> xTestTensor)
        {
            if (!Enumerable.SequenceEqual(xTrainTensor.Shape.Skip(1), xValTensor.Shape.Skip(1)) ||
                !Enumerable.SequenceEqual(xTrainTensor.Shape.Skip(1), xTestTensor.Shape.Skip(1)))
                throw new ArgumentException("All input tensors must have the same shape (except for the first dimension).");
        }
        else
        {
            throw new ArgumentException("Inconsistent input types across datasets.");
        }
    }
}
