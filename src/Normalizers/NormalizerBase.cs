namespace AiDotNet.Normalizers;

/// <summary>
/// Base class for normalizers that provides common functionality and implements the INormalizer interface.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The type of input data structure.</typeparam>
/// <typeparam name="TOutput">The type of output data structure.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This base class provides shared functionality for all normalization methods.
/// Normalization transforms data into a standard range, which helps machine learning algorithms
/// process features fairly regardless of their original scales.</para>
/// </remarks>
public abstract class NormalizerBase<T, TInput, TOutput> : INormalizer<T, TInput, TOutput>
{
    /// <summary>
    /// The numeric operations provider for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Initializes a new instance of the NormalizerBase class.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor initializes the numeric operations provider,
    /// which enables mathematical operations on different types of numbers (float, double, etc.).</para>
    /// </remarks>
    protected NormalizerBase()
    {
        NumOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Normalizes output data to a standard range.
    /// </summary>
    /// <param name="data">The data to normalize.</param>
    /// <returns>A tuple containing the normalized data and the normalization parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method converts your data to a standard scale.
    /// It returns both the normalized data and information needed to convert back later.</para>
    /// </remarks>
    public abstract (TOutput, NormalizationParameters<T>) NormalizeOutput(TOutput data);

    /// <summary>
    /// Normalizes input data to a standard range.
    /// </summary>
    /// <param name="data">The data to normalize.</param>
    /// <returns>A tuple containing the normalized data and a list of normalization parameters for each feature.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method normalizes multiple features at once.
    /// Each feature (column) is normalized separately, and parameters for each are returned.</para>
    /// </remarks>
    public abstract (TInput, List<NormalizationParameters<T>>) NormalizeInput(TInput data);

    /// <summary>
    /// Reverses the normalization of data using the original normalization parameters.
    /// </summary>
    /// <param name="data">The normalized data to denormalize.</param>
    /// <param name="parameters">The normalization parameters used during normalization.</param>
    /// <returns>The denormalized data in its original scale.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method converts normalized values back to their original scale.</para>
    /// </remarks>
    public abstract TOutput Denormalize(TOutput data, NormalizationParameters<T> parameters);

    /// <summary>
    /// Denormalizes model coefficients to make them applicable to non-normalized input data.
    /// </summary>
    /// <param name="coefficients">The model coefficients from a model trained on normalized data.</param>
    /// <param name="xParams">The normalization parameters used for the input features.</param>
    /// <param name="yParams">The normalization parameters used for the target variable.</param>
    /// <returns>Denormalized coefficients for use with original, non-normalized data.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This adjusts model weights to work with your original data.</para>
    /// </remarks>
    public abstract TOutput Denormalize(TOutput coefficients, List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams);

    /// <summary>
    /// Calculates the denormalized Y-intercept (constant term) for a linear model.
    /// </summary>
    /// <param name="xMatrix">The original input feature matrix.</param>
    /// <param name="y">The original target vector.</param>
    /// <param name="coefficients">The model coefficients.</param>
    /// <param name="xParams">The normalization parameters for the input features.</param>
    /// <param name="yParams">The normalization parameters for the target variable.</param>
    /// <returns>The denormalized Y-intercept for use with non-normalized data.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This calculates the starting point for your model's predictions.</para>
    /// </remarks>
    public abstract T Denormalize(TInput xMatrix, TOutput y, TOutput coefficients,
        List<NormalizationParameters<T>> xParams, NormalizationParameters<T> yParams);
}
