namespace AiDotNet.Helpers;

/// <summary>
/// Provides numerical stability utilities for safe mathematical operations in machine learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Machine learning algorithms often deal with very small or very large numbers,
/// which can cause numerical issues like:
/// - Division by zero
/// - Log of zero or negative numbers
/// - NaN (Not a Number) values appearing in calculations
/// - Infinity values from overflow
///
/// This helper provides safe versions of common operations that avoid these problems.
/// </para>
/// </remarks>
public static class NumericalStabilityHelper
{
    /// <summary>
    /// Default epsilon value for numerical stability (1e-7 for float precision).
    /// </summary>
    public const double DefaultEpsilon = 1e-7;

    /// <summary>
    /// Smaller epsilon for double precision operations (1e-15).
    /// </summary>
    public const double SmallEpsilon = 1e-15;

    /// <summary>
    /// Larger epsilon for less sensitive operations (1e-5).
    /// </summary>
    public const double LargeEpsilon = 1e-5;

    /// <summary>
    /// Gets a type-appropriate epsilon value for the numeric type T.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="epsilon">Optional custom epsilon. If null, uses type-appropriate default.</param>
    /// <returns>The epsilon value converted to type T.</returns>
    public static T GetEpsilon<T>(double? epsilon = null)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        double eps = epsilon ?? DefaultEpsilon;
        return numOps.FromDouble(eps);
    }

    /// <summary>
    /// Computes the natural logarithm safely, avoiding log(0) and log(negative).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="value">The value to compute log of.</param>
    /// <param name="epsilon">Small value to add for numerical stability. Defaults to 1e-7.</param>
    /// <returns>log(max(value, epsilon))</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The logarithm of zero is negative infinity, and log of negative
    /// numbers is undefined. This method ensures we always compute log of a small positive number
    /// at minimum, preventing NaN or -Infinity in your calculations.
    /// </para>
    /// </remarks>
    public static T SafeLog<T>(T value, double epsilon = DefaultEpsilon)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        T eps = numOps.FromDouble(epsilon);
        T safeValue = numOps.LessThan(value, eps) ? eps : value;
        return numOps.Log(safeValue);
    }

    /// <summary>
    /// Performs safe division, avoiding division by zero.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="numerator">The numerator.</param>
    /// <param name="denominator">The denominator.</param>
    /// <param name="epsilon">Small value to add to denominator for stability. Defaults to 1e-7.</param>
    /// <returns>numerator / (denominator + epsilon) if denominator is near zero, else numerator / denominator.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Division by zero results in infinity or NaN. This method adds
    /// a tiny value to very small denominators to prevent this while minimally affecting the result.
    /// </para>
    /// </remarks>
    public static T SafeDiv<T>(T numerator, T denominator, double epsilon = DefaultEpsilon)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        T eps = numOps.FromDouble(epsilon);
        T absDenom = numOps.Abs(denominator);

        if (numOps.LessThan(absDenom, eps))
        {
            // Add epsilon with the sign of the original denominator
            T sign = numOps.LessThan(denominator, numOps.Zero) ? numOps.FromDouble(-1) : numOps.One;
            denominator = numOps.Multiply(sign, eps);
        }

        return numOps.Divide(numerator, denominator);
    }

    /// <summary>
    /// Computes square root safely, ensuring non-negative input.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="value">The value to compute square root of.</param>
    /// <param name="epsilon">Small value to ensure positive input. Defaults to 1e-7.</param>
    /// <returns>sqrt(max(value, epsilon))</returns>
    public static T SafeSqrt<T>(T value, double epsilon = DefaultEpsilon)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        T eps = numOps.FromDouble(epsilon);
        T safeValue = numOps.LessThan(value, eps) ? eps : value;
        return numOps.Sqrt(safeValue);
    }

    /// <summary>
    /// Clamps a value to valid probability range [epsilon, 1-epsilon].
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="probability">The probability value to clamp.</param>
    /// <param name="epsilon">Small value for bounds. Defaults to 1e-7.</param>
    /// <returns>The clamped probability.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Probabilities should be between 0 and 1, but for numerical
    /// stability (especially when taking log of probabilities), we clamp to [epsilon, 1-epsilon]
    /// to avoid log(0) and log(1) issues.
    /// </para>
    /// </remarks>
    public static T ClampProbability<T>(T probability, double epsilon = DefaultEpsilon)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        T eps = numOps.FromDouble(epsilon);
        T oneMinusEps = numOps.Subtract(numOps.One, eps);
        return MathHelper.Clamp(probability, eps, oneMinusEps);
    }

    /// <summary>
    /// Computes safe log of a probability (clamps first, then takes log).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="probability">The probability value.</param>
    /// <param name="epsilon">Small value for stability. Defaults to 1e-7.</param>
    /// <returns>log(clamp(probability, epsilon, 1-epsilon))</returns>
    public static T SafeLogProbability<T>(T probability, double epsilon = DefaultEpsilon)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        T clampedProb = ClampProbability(probability, epsilon);
        return numOps.Log(clampedProb);
    }

    /// <summary>
    /// Checks if a value is NaN (Not a Number).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="value">The value to check.</param>
    /// <returns>True if the value is NaN.</returns>
    public static bool IsNaN<T>(T value)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return numOps.IsNaN(value);
    }

    /// <summary>
    /// Checks if a value is infinite (positive or negative infinity).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="value">The value to check.</param>
    /// <returns>True if the value is infinite.</returns>
    public static bool IsInfinity<T>(T value)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return numOps.IsInfinity(value);
    }

    /// <summary>
    /// Checks if a value is finite (not NaN and not infinite).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="value">The value to check.</param>
    /// <returns>True if the value is finite.</returns>
    public static bool IsFinite<T>(T value)
    {
        return !IsNaN(value) && !IsInfinity(value);
    }

    /// <summary>
    /// Checks if a vector contains any NaN values.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="vector">The vector to check.</param>
    /// <returns>True if any element is NaN.</returns>
    public static bool ContainsNaN<T>(Vector<T> vector)
    {
        if (vector == null) return false;

        for (int i = 0; i < vector.Length; i++)
        {
            if (IsNaN(vector[i])) return true;
        }
        return false;
    }

    /// <summary>
    /// Checks if a vector contains any infinite values.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="vector">The vector to check.</param>
    /// <returns>True if any element is infinite.</returns>
    public static bool ContainsInfinity<T>(Vector<T> vector)
    {
        if (vector == null) return false;

        for (int i = 0; i < vector.Length; i++)
        {
            if (IsInfinity(vector[i])) return true;
        }
        return false;
    }

    /// <summary>
    /// Checks if a vector contains any non-finite values (NaN or infinite).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="vector">The vector to check.</param>
    /// <returns>True if any element is non-finite.</returns>
    public static bool ContainsNonFinite<T>(Vector<T> vector)
    {
        return ContainsNaN(vector) || ContainsInfinity(vector);
    }

    /// <summary>
    /// Checks if a tensor contains any NaN values.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="tensor">The tensor to check.</param>
    /// <returns>True if any element is NaN.</returns>
    public static bool ContainsNaN<T>(Tensor<T> tensor)
    {
        if (tensor == null) return false;

        for (int i = 0; i < tensor.Length; i++)
        {
            if (IsNaN(tensor[i])) return true;
        }
        return false;
    }

    /// <summary>
    /// Checks if a tensor contains any infinite values.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="tensor">The tensor to check.</param>
    /// <returns>True if any element is infinite.</returns>
    public static bool ContainsInfinity<T>(Tensor<T> tensor)
    {
        if (tensor == null) return false;

        for (int i = 0; i < tensor.Length; i++)
        {
            if (IsInfinity(tensor[i])) return true;
        }
        return false;
    }

    /// <summary>
    /// Checks if a tensor contains any non-finite values (NaN or infinite).
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="tensor">The tensor to check.</param>
    /// <returns>True if any element is non-finite.</returns>
    public static bool ContainsNonFinite<T>(Tensor<T> tensor)
    {
        return ContainsNaN(tensor) || ContainsInfinity(tensor);
    }

    /// <summary>
    /// Replaces NaN values in a vector with a specified replacement value.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="vector">The vector to process.</param>
    /// <param name="replacement">The value to replace NaN with (defaults to zero).</param>
    /// <returns>A new vector with NaN values replaced.</returns>
    public static Vector<T>? ReplaceNaN<T>(Vector<T>? vector, T? replacement = default)
    {
        if (vector == null) return null;

        var numOps = MathHelper.GetNumericOperations<T>();
        T replaceValue = replacement ?? numOps.Zero;

        var result = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = IsNaN(vector[i]) ? replaceValue : vector[i];
        }
        return result;
    }

    /// <summary>
    /// Replaces infinite values in a vector with a specified replacement value.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="vector">The vector to process.</param>
    /// <param name="replacement">The value to replace infinity with (defaults to zero).</param>
    /// <returns>A new vector with infinite values replaced.</returns>
    public static Vector<T>? ReplaceInfinity<T>(Vector<T>? vector, T? replacement = default)
    {
        if (vector == null) return null;

        var numOps = MathHelper.GetNumericOperations<T>();
        T replaceValue = replacement ?? numOps.Zero;

        var result = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = IsInfinity(vector[i]) ? replaceValue : vector[i];
        }
        return result;
    }

    /// <summary>
    /// Replaces all non-finite values (NaN and infinity) in a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="vector">The vector to process.</param>
    /// <param name="replacement">The value to replace non-finite values with (defaults to zero).</param>
    /// <returns>A new vector with non-finite values replaced.</returns>
    public static Vector<T>? ReplaceNonFinite<T>(Vector<T>? vector, T? replacement = default)
    {
        if (vector == null) return null;

        var numOps = MathHelper.GetNumericOperations<T>();
        T replaceValue = replacement ?? numOps.Zero;

        var result = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            result[i] = IsFinite(vector[i]) ? vector[i] : replaceValue;
        }
        return result;
    }

    /// <summary>
    /// Computes softmax with numerical stability using the log-sum-exp trick.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="logits">The input logits.</param>
    /// <returns>Softmax probabilities.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Softmax converts a vector of numbers into probabilities.
    /// The standard formula exp(x_i) / sum(exp(x_j)) can overflow for large values.
    /// This implementation subtracts the maximum value first to prevent overflow.
    /// </para>
    /// </remarks>
    public static Vector<T>? StableSoftmax<T>(Vector<T>? logits)
    {
        if (logits == null || logits.Length == 0)
            return logits;

        var numOps = MathHelper.GetNumericOperations<T>();

        // Find max for numerical stability
        T maxVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (numOps.GreaterThan(logits[i], maxVal))
                maxVal = logits[i];
        }

        // Compute exp(x - max) and sum
        var expValues = new Vector<T>(logits.Length);
        T sum = numOps.Zero;
        for (int i = 0; i < logits.Length; i++)
        {
            expValues[i] = numOps.Exp(numOps.Subtract(logits[i], maxVal));
            sum = numOps.Add(sum, expValues[i]);
        }

        // Normalize with epsilon protection
        T eps = numOps.FromDouble(DefaultEpsilon);
        if (numOps.LessThan(sum, eps))
            sum = eps;

        var result = new Vector<T>(logits.Length);
        for (int i = 0; i < logits.Length; i++)
        {
            result[i] = numOps.Divide(expValues[i], sum);
        }

        return result;
    }

    /// <summary>
    /// Computes log-softmax with numerical stability.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="logits">The input logits.</param>
    /// <returns>Log-softmax values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Log-softmax is log(softmax(x)), which is more numerically
    /// stable than computing softmax first and then taking the log. It's commonly used
    /// in cross-entropy loss calculations.
    /// </para>
    /// </remarks>
    public static Vector<T>? StableLogSoftmax<T>(Vector<T>? logits)
    {
        if (logits == null || logits.Length == 0)
            return logits;

        var numOps = MathHelper.GetNumericOperations<T>();

        // Find max for numerical stability
        T maxVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (numOps.GreaterThan(logits[i], maxVal))
                maxVal = logits[i];
        }

        // Compute log-sum-exp
        T sumExp = numOps.Zero;
        for (int i = 0; i < logits.Length; i++)
        {
            sumExp = numOps.Add(sumExp, numOps.Exp(numOps.Subtract(logits[i], maxVal)));
        }

        T logSumExp = numOps.Add(maxVal, SafeLog(sumExp));

        // log_softmax(x_i) = x_i - log_sum_exp
        var result = new Vector<T>(logits.Length);
        for (int i = 0; i < logits.Length; i++)
        {
            result[i] = numOps.Subtract(logits[i], logSumExp);
        }

        return result;
    }

    /// <summary>
    /// Counts the number of NaN values in a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="vector">The vector to check.</param>
    /// <returns>The count of NaN values.</returns>
    public static int CountNaN<T>(Vector<T> vector)
    {
        if (vector == null) return 0;

        int count = 0;
        for (int i = 0; i < vector.Length; i++)
        {
            if (IsNaN(vector[i])) count++;
        }
        return count;
    }

    /// <summary>
    /// Counts the number of infinite values in a vector.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="vector">The vector to check.</param>
    /// <returns>The count of infinite values.</returns>
    public static int CountInfinity<T>(Vector<T> vector)
    {
        if (vector == null) return 0;

        int count = 0;
        for (int i = 0; i < vector.Length; i++)
        {
            if (IsInfinity(vector[i])) count++;
        }
        return count;
    }

    /// <summary>
    /// Asserts that a value is finite, throwing if not.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="value">The value to check.</param>
    /// <param name="paramName">Name of the parameter for the exception message.</param>
    /// <exception cref="ArgumentException">Thrown if the value is not finite.</exception>
    public static void AssertFinite<T>(T value, string paramName = "value")
    {
        if (IsNaN(value))
            throw new ArgumentException($"Value is NaN", paramName);
        if (IsInfinity(value))
            throw new ArgumentException($"Value is infinite", paramName);
    }

    /// <summary>
    /// Asserts that a vector contains only finite values.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="vector">The vector to check.</param>
    /// <param name="paramName">Name of the parameter for the exception message.</param>
    /// <exception cref="ArgumentException">Thrown if the vector contains non-finite values.</exception>
    public static void AssertFinite<T>(Vector<T> vector, string paramName = "vector")
    {
        if (vector == null) return;

        int nanCount = CountNaN(vector);
        int infCount = CountInfinity(vector);

        if (nanCount > 0 || infCount > 0)
        {
            throw new ArgumentException(
                $"Vector contains {nanCount} NaN and {infCount} infinite values",
                paramName);
        }
    }

    /// <summary>
    /// Asserts that a tensor contains only finite values.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="tensor">The tensor to check.</param>
    /// <param name="paramName">Name of the parameter for the exception message.</param>
    /// <exception cref="ArgumentException">Thrown if the tensor contains non-finite values.</exception>
    public static void AssertFinite<T>(Tensor<T> tensor, string paramName = "tensor")
    {
        if (tensor == null) return;

        if (ContainsNaN(tensor))
            throw new ArgumentException($"Tensor contains NaN values", paramName);
        if (ContainsInfinity(tensor))
            throw new ArgumentException($"Tensor contains infinite values", paramName);
    }
}
