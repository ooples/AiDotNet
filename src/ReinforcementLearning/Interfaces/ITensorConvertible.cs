using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Interfaces
{
    /// <summary>
    /// Represents a type that can be converted to a tensor.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public interface ITensorConvertible<T>
    {
        /// <summary>
        /// Converts the object to a tensor.
        /// </summary>
        /// <returns>The tensor representation of the object.</returns>
        Tensor<T> ToTensor();
    }
}