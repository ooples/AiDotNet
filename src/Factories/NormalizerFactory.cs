namespace AiDotNet.Factories;

/// <summary>
/// A factory class that creates data normalizers for preprocessing machine learning inputs.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Normalization is the process of transforming data to a standard scale, 
/// which helps machine learning algorithms perform better. Think of it like converting different 
/// currencies to a single currency so they can be compared fairly.
/// </para>
/// <para>
/// This factory helps you create different types of normalizers without needing to know their 
/// internal implementation details. Think of it like ordering a specific tool from a catalog - 
/// you just specify what you need, and the factory provides it.
/// </para>
/// </remarks>
public class NormalizerFactory<T, TInput, TOutput>
{
    /// <summary>
    /// Provides operations for numeric calculations with type T.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is a helper object that knows how to perform math operations 
    /// on the specific number type you're using (like float or double).
    /// </remarks>
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Creates a normalizer of the specified method.
    /// </summary>
    /// <param name="method">The normalization method to use.</param>
    /// <param name="lpNormP">Optional parameter for Lp-norm normalization, defaults to 2 (Euclidean norm).</param>
    /// <returns>An implementation of INormalizer<T, TInput, TOutput> for the specified normalization method.</returns>
    /// <exception cref="ArgumentException">Thrown when an unsupported normalization method is specified.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Different normalization methods transform your data in different ways. 
    /// Choose the method that works best for your specific machine learning task.
    /// </para>
    /// <para>
    /// Available normalization methods include:
    /// <list type="bullet">
    /// <item><description>None: No normalization is applied (data remains unchanged).</description></item>
    /// <item><description>MinMax: Scales data to a specific range, typically [0,1].</description></item>
    /// <item><description>ZScore: Transforms data to have a mean of 0 and standard deviation of 1.</description></item>
    /// <item><description>RobustScaling: Similar to MinMax but uses percentiles instead of min/max, making it less sensitive to outliers.</description></item>
    /// <item><description>Decimal: Scales values by dividing by powers of 10 to bring them to a decimal range.</description></item>
    /// <item><description>Binning: Groups continuous data into discrete bins or categories.</description></item>
    /// <item><description>MeanVariance: Adjusts data to have a specific mean and variance.</description></item>
    /// <item><description>LogMeanVariance: Applies logarithmic transformation before mean-variance normalization.</description></item>
    /// <item><description>GlobalContrast: Adjusts the contrast of the entire dataset.</description></item>
    /// <item><description>LpNorm: Scales data using the Lp-norm (p=2 is the Euclidean norm).</description></item>
    /// <item><description>Log: Applies a logarithmic transformation to compress the range of values.</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public static INormalizer<T, TInput, TOutput> CreateNormalizer(NormalizationMethod method, T? lpNormP = default)
    {
        return method switch
        {
            NormalizationMethod.None => new NoNormalizer<T, TInput, TOutput>(),
            NormalizationMethod.MinMax => new MinMaxNormalizer<T, TInput, TOutput>(),
            NormalizationMethod.ZScore => new ZScoreNormalizer<T, TInput, TOutput>(),
            NormalizationMethod.RobustScaling => new RobustScalingNormalizer<T, TInput, TOutput>(),
            NormalizationMethod.Decimal => new DecimalNormalizer<T, TInput, TOutput>(),
            NormalizationMethod.Binning => new BinningNormalizer<T, TInput, TOutput>(),
            NormalizationMethod.MeanVariance => new MeanVarianceNormalizer<T, TInput, TOutput>(),
            NormalizationMethod.LogMeanVariance => new LogMeanVarianceNormalizer<T, TInput, TOutput>(),
            NormalizationMethod.GlobalContrast => new GlobalContrastNormalizer<T, TInput, TOutput>(),
            NormalizationMethod.LpNorm => new LpNormNormalizer<T, TInput, TOutput>(lpNormP ?? _numOps.FromDouble(2)),
            NormalizationMethod.Log => new LogNormalizer<T, TInput, TOutput>(),
            _ => throw new ArgumentException($"Unsupported normalization method: {method}")
        };
    }
}
