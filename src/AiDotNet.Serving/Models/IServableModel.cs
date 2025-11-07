using AiDotNet.LinearAlgebra;

namespace AiDotNet.Serving.Models;

/// <summary>
/// Interface for models that can be served via the REST API.
/// Models must implement this interface to support prediction operations.
/// </summary>
/// <typeparam name="T">The numeric type used by the model (e.g., double, float, decimal)</typeparam>
public interface IServableModel<T> where T : struct
{
    /// <summary>
    /// Performs prediction on a single input vector.
    /// </summary>
    /// <param name="input">The input features as a vector</param>
    /// <returns>The prediction result as a vector</returns>
    Vector<T> Predict(Vector<T> input);

    /// <summary>
    /// Performs batch prediction on multiple input vectors.
    /// This method should be optimized for processing multiple inputs at once.
    /// </summary>
    /// <param name="inputs">The matrix where each row is an input sample</param>
    /// <returns>The prediction results as a matrix where each row corresponds to an input</returns>
    Matrix<T> PredictBatch(Matrix<T> inputs);

    /// <summary>
    /// Gets the name of the model.
    /// </summary>
    string ModelName { get; }

    /// <summary>
    /// Gets the expected number of input features.
    /// </summary>
    int InputDimension { get; }

    /// <summary>
    /// Gets the number of output dimensions.
    /// </summary>
    int OutputDimension { get; }
}
