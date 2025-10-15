namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for models that have optimizable parameters.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface IParameterizable<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the parameters that can be optimized.
    /// </summary>
    Vector<T> GetParameters();

    /// <summary>
    /// Sets the parameters of the model.
    /// </summary>
    /// <param name="parameters">The parameters to set.</param>
    void SetParameters(Vector<T> parameters);

    /// <summary>
    /// Creates a new instance with the specified parameters.
    /// </summary>
    IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters);
}