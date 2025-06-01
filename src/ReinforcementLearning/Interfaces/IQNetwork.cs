using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.ReinforcementLearning.Interfaces;

/// <summary>
/// Interface for Q-value neural networks used in reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TOutput">The output type (Vector<double>{T} for standard Q-networks, Tensor<double>{T} for distributional).</typeparam>
public interface IQNetwork<T, TOutput> : IFullModel<T, Tensor<T>, TOutput>
{
    /// <summary>
    /// Gets the size of the state space.
    /// </summary>
    int StateSize { get; }

    /// <summary>
    /// Gets the size of the action space.
    /// </summary>
    int ActionSize { get; }

    /// <summary>
    /// Gets a value indicating whether this network uses a dueling architecture.
    /// </summary>
    bool IsDueling { get; }

    /// <summary>
    /// Performs backward propagation through the network.
    /// </summary>
    /// <param name="losses">The losses to backpropagate.</param>
    /// <param name="optimizer">The optimizer to use for updating parameters.</param>
    void Backward(Vector<T> losses, IOptimizer<T, Tensor<T>, Tensor<T>> optimizer);

    /// <summary>
    /// Copies parameters from another Q-network.
    /// </summary>
    /// <param name="other">The source network to copy from.</param>
    /// <param name="tau">The interpolation factor (1.0 for full copy, less for soft update).</param>
    void CopyFrom(IQNetwork<T, TOutput> other, T tau);

    /// <summary>
    /// Gets the underlying neural network.
    /// </summary>
    INeuralNetworkModel<T> GetUnderlyingNetwork();

    /// <summary>
    /// Sets the training mode of the network.
    /// </summary>
    /// <param name="isTraining">Whether the network should be in training mode.</param>
    void SetTrainingMode(bool isTraining);
}