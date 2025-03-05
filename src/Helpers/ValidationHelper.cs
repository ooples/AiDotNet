using System.Diagnostics;

namespace AiDotNet.Helpers;

public static class ValidationHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public static void ValidateInputData(Matrix<T> x, Vector<T> y)
    {
        ValidateMatrixVectorPair(x, y, "Input");
    }

    /// <summary>
    /// Gets information about the calling method.
    /// </summary>
    /// <param name="skipFrames">Number of frames to skip in the stack trace.</param>
    /// <returns>A tuple containing the component name and operation name.</returns>
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
            if (NumOps.LessThan(y[i], NumOps.Zero) || !MathHelper.IsInteger(y[i]))
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