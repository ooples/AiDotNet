using System.Diagnostics;

namespace AiDotNet.Helpers;

/// <summary>
/// Provides validation methods for AI model inputs and parameters.
/// </summary>
/// <typeparam name="T">The numeric type used in calculations (e.g., double, float).</typeparam>
/// <remarks>
/// For Beginners: This helper class ensures that the data you provide to AI models is valid and properly formatted.
/// Think of it as a quality control checkpoint that prevents errors before they happen by checking that your
/// data meets all the requirements needed for successful model training and prediction.
/// </remarks>
public static class ValidationHelper<T>
{
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Validates that input data matrices and vectors are properly formatted for model training.
    /// </summary>
    /// <param name="x">The feature matrix containing input variables.</param>
    /// <param name="y">The target vector containing output values to predict.</param>
    /// <remarks>
    /// For Beginners: This method checks that your input data (x) and output data (y) are compatible.
    /// The input data (x) is a matrix where each row represents one data point and each column represents
    /// one feature or characteristic. The output data (y) is a vector where each element is the target value
    /// you want to predict for the corresponding row in x. This method ensures they have matching dimensions.
    /// </remarks>
    public static void ValidateInputData(Matrix<T> x, Vector<T> y)
    {
        ValidateMatrixVectorPair(x, y, "Input");
    }

    /// <summary>
    /// Gets information about the calling method.
    /// </summary>
    /// <param name="skipFrames">Number of frames to skip in the stack trace.</param>
    /// <returns>A tuple containing the component name and operation name.</returns>
    /// <remarks>
    /// For Beginners: This method identifies which part of the code called a validation function.
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
    /// For Beginners: This method helps create more informative error messages by identifying
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
    /// <param name="inputData">The optimization input data containing training, validation, and test datasets.</param>
    /// <remarks>
    /// For Beginners: When training AI models, we typically split our data into three sets:
    /// 1. Training data - used to teach the model patterns (like studying for a test)
    /// 2. Validation data - used to tune the model (like practice tests)
    /// 3. Test data - used to evaluate the final model (like the final exam)
    /// 
    /// This method checks that all three datasets are properly formatted and compatible with each other.
    /// It ensures they have the same number of features (columns) and appropriate dimensions.
    /// </remarks>
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

    /// <summary>
    /// Validates that data is appropriate for Poisson regression.
    /// </summary>
    /// <param name="y">The target vector containing output values to predict.</param>
    /// <remarks>
    /// For Beginners: Poisson regression is a special type of model used when predicting counts
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

    /// <summary>
    /// Validates that a matrix and vector pair have compatible dimensions and are not null or empty.
    /// </summary>
    /// <param name="x">The feature matrix containing input variables.</param>
    /// <param name="y">The target vector containing output values to predict.</param>
    /// <param name="datasetName">The name of the dataset being validated (e.g., "Training", "Test").</param>
    /// <remarks>
    /// For Beginners: This method ensures that your input data (x) and output data (y) are compatible.
    /// It checks that:
    /// 1. Neither is null (missing entirely)
    /// 2. The number of rows in x matches the length of y (each input row has a corresponding output value)
    /// 3. The data isn't empty (has at least one row and column)
    /// 
    /// This is like making sure you have the same number of questions and answers on a test.
    /// </remarks>
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