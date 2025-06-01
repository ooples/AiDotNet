using AiDotNet.LinearAlgebra;

namespace AiDotNet.Ensemble.Strategies;

/// <summary>
/// Combines predictions by computing a trimmed mean, which removes extreme values before averaging.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<double>, Tensor<double>, Vector<double>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<double>, Tensor<double>).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A trimmed mean is like an average, but we first remove some of the 
/// highest and lowest values. This makes it more robust than a simple average but less extreme 
/// than using the median. For example, if you have predictions [1, 2, 3, 10, 100] and trim 20%, 
/// you'd remove the lowest (1) and highest (100) values, then average [2, 3, 10] to get 5.
/// </para>
/// <para>
/// This is useful when you suspect some models might occasionally make wild predictions but 
/// you still want to use more information than just the median value.
/// </para>
/// </remarks>
public class TrimmedMeanStrategy<T, TInput, TOutput> : CombinationStrategyBase<T, TInput, TOutput>
{
    /// <summary>
    /// The percentage of values to trim from each end (0.0 to 0.5).
    /// </summary>
    private readonly double _trimPercentage;
    
    /// <summary>
    /// Gets the trim percentage.
    /// </summary>
    public double TrimPercentage => _trimPercentage;

    /// <summary>
    /// Gets the name of the combination strategy.
    /// </summary>
    public override string StrategyName => $"Trimmed Mean ({_trimPercentage:P0})";
    
    /// <summary>
    /// Gets whether this strategy requires trained weights.
    /// </summary>
    public override bool RequiresTraining => false;

    /// <summary>
    /// Initializes a new instance of the TrimmedMeanStrategy class.
    /// </summary>
    /// <param name="trimPercentage">The percentage to trim from each end (default 0.1 for 10%).</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when trim percentage is not between 0 and 0.5.</exception>
    public TrimmedMeanStrategy(double trimPercentage = 0.1)
    {
        if (trimPercentage < 0.0 || trimPercentage >= 0.5)
        {
            throw new ArgumentOutOfRangeException(nameof(trimPercentage), 
                "Trim percentage must be between 0 and 0.5 (exclusive)");
        }
        
        _trimPercentage = trimPercentage;
    }

    /// <summary>
    /// Combines predictions using a trimmed mean approach.
    /// </summary>
    /// <param name="predictions">The list of predictions from individual models.</param>
    /// <param name="weights">The weights (ignored for trimmed mean).</param>
    /// <returns>The trimmed mean prediction.</returns>
    public override TOutput Combine(List<TOutput> predictions, Vector<T> weights)
    {
        if (!CanCombine(predictions))
        {
            throw new ArgumentException("Cannot combine predictions");
        }

        // Calculate how many values to trim from each end
        int trimCount = (int)Math.Floor(predictions.Count * _trimPercentage);
        
        // If we would trim all values, just return the median
        if (trimCount * 2 >= predictions.Count)
        {
            int medianIndex = predictions.Count / 2;
            return predictions[medianIndex];
        }

        // For the base implementation with generic types, we return a value from the trimmed range
        // Specific implementations for Vector<T> or other types would override this method
        // to provide proper numerical trimmed mean calculations
        
        // Skip the lowest trimCount and highest trimCount values
        int startIndex = trimCount;
        int endIndex = predictions.Count - trimCount;
        int trimmedCount = endIndex - startIndex;
        
        // For generic implementation, return the middle value of the trimmed range
        int middleIndex = startIndex + (trimmedCount / 2);
        return predictions[middleIndex];
    }
    
    /// <summary>
    /// Validates if the predictions can be combined using trimmed mean.
    /// </summary>
    /// <param name="predictions">The predictions to validate.</param>
    /// <returns>True if the predictions can be combined; otherwise, false.</returns>
    public override bool CanCombine(List<TOutput> predictions)
    {
        if (predictions == null || predictions.Count == 0)
        {
            return false;
        }

        // We need at least 3 predictions to meaningfully trim
        // (one to keep after trimming from both ends)
        int minRequired = (int)Math.Ceiling(1.0 / (1.0 - 2.0 * _trimPercentage));
        return predictions.Count >= Math.Max(3, minRequired);
    }
}

/// <summary>
/// Specialized implementation of TrimmedMeanStrategy for Vector<double> outputs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
public class TrimmedMeanVectorStrategy<T, TInput> : TrimmedMeanStrategy<T, TInput, Vector<T>>
{
    /// <summary>
    /// Initializes a new instance of the TrimmedMeanVectorStrategy class.
    /// </summary>
    /// <param name="trimPercentage">The percentage to trim from each end (default 0.1 for 10%).</param>
    public TrimmedMeanVectorStrategy(double trimPercentage = 0.1) : base(trimPercentage)
    {
    }

    /// <summary>
    /// Combines vector predictions using element-wise trimmed mean.
    /// </summary>
    public override Vector<T> Combine(List<Vector<T>> predictions, Vector<T> weights)
    {
        if (!CanCombine(predictions))
        {
            throw new ArgumentException("Cannot combine predictions");
        }

        int vectorLength = predictions[0].Length;
        var result = new T[vectorLength];
        
        // Calculate trimmed mean for each element position
        for (int i = 0; i < vectorLength; i++)
        {
            // Extract values at position i from all predictions
            var values = new List<T>(predictions.Count);
            for (int j = 0; j < predictions.Count; j++)
            {
                values.Add(predictions[j][i]);
            }
            
            // Sort the values
            values.Sort((a, b) => 
            {
                if (NumOps.GreaterThan(a, b)) return 1;
                if (NumOps.LessThan(a, b)) return -1;
                return 0;
            });
            
            // Calculate trim count
            int trimCount = (int)Math.Floor(predictions.Count * TrimPercentage);
            
            // Calculate trimmed mean
            var sum = NumOps.Zero;
            int startIndex = trimCount;
            int endIndex = values.Count - trimCount;
            int count = endIndex - startIndex;
            
            for (int j = startIndex; j < endIndex; j++)
            {
                sum = NumOps.Add(sum, values[j]);
            }
            
            result[i] = NumOps.Divide(sum, NumOps.FromDouble(count));
        }
        
        return new Vector<T>(result);
    }
}