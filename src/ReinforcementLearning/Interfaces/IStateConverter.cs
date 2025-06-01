using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Interfaces
{
    /// <summary>
    /// Interface for state converters that transform states into tensor representation.
    /// </summary>
    /// <typeparam name="TState">The type of state to convert</typeparam>
    /// <typeparam name="T">The numeric type for tensor values</typeparam>
    public interface IStateConverter<TState, T>
    {
        /// <summary>
        /// Converts a single state to tensor representation
        /// </summary>
        /// <param name="state">The state to convert</param>
        /// <returns>Tensor<double> representation of the state</returns>
        Tensor<T> ConvertState(TState state);
        
        /// <summary>
        /// Converts an array of states to a batch tensor
        /// </summary>
        /// <param name="states">Array of states to convert</param>
        /// <returns>Batch tensor representation of states</returns>
        Tensor<T> ConvertStates(TState[] states);
    }
}