using System;
using System.IO;
using AiDotNet.NeuralNetworks;
using AiDotNet.Interfaces;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.ReinforcementLearning.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;

namespace AiDotNet.Extensions
{
    /// <summary>
    /// Extension methods for neural network operations.
    /// </summary>
    public static class NeuralNetworkExtensions
    {
        /// <summary>
        /// Saves a neural network model to a file.
        /// </summary>
        /// <typeparam name="T">The numeric type used by the network.</typeparam>
        /// <param name="network">The neural network to save.</param>
        /// <param name="filePath">The file path to save to.</param>
        public static void SaveModel<T>(this NeuralNetwork<T> network, string filePath)
        {
            if (network == null)
                throw new ArgumentNullException(nameof(network));
            
            if (string.IsNullOrEmpty(filePath))
                throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
            
            // Ensure the directory exists
            var directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }
            
            // Add .bin extension if not present
            if (!filePath.EndsWith(".bin", StringComparison.OrdinalIgnoreCase))
            {
                filePath += ".bin";
            }
            
            // Serialize and save
            var data = network.Serialize();
            File.WriteAllBytes(filePath, data);
        }
        
        /// <summary>
        /// Loads a neural network model from a file.
        /// </summary>
        /// <typeparam name="T">The numeric type used by the network.</typeparam>
        /// <param name="network">The neural network to load into (not currently used - placeholder for future implementation).</param>
        /// <param name="filePath">The file path to load from.</param>
        /// <remarks>
        /// Note: This is a placeholder implementation. Proper deserialization requires recreating the network
        /// with the same architecture and then loading the parameters.
        /// </remarks>
        public static void LoadModel<T>(this NeuralNetwork<T> network, string filePath)
        {
            if (network == null)
                throw new ArgumentNullException(nameof(network));
            
            if (string.IsNullOrEmpty(filePath))
                throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
            
            // Add .bin extension if not present
            if (!filePath.EndsWith(".bin", StringComparison.OrdinalIgnoreCase))
            {
                filePath += ".bin";
            }
            
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"Model file not found: {filePath}");
            
            // Read the serialized data
            byte[] data = File.ReadAllBytes(filePath);
            
            // Deserialize the network
            network.Deserialize(data);
        }
        
        /// <summary>
        /// Performs backward propagation with a given loss gradient.
        /// </summary>
        /// <typeparam name="T">The numeric type used by the network.</typeparam>
        /// <param name="network">The neural network.</param>
        /// <param name="lossGradient">The gradient of the loss with respect to the network output.</param>
        /// <remarks>
        /// This method assumes the loss gradient has been computed externally and 
        /// propagates it backward through the network.
        /// </remarks>
        public static void Backward<T>(this NeuralNetwork<T> network, T lossGradient)
        {
            // Get the output shape from the architecture
            var outputShape = network.Architecture.GetOutputShape();
            
            // Create a tensor from the scalar loss gradient
            // For a scalar loss, we typically have a gradient of 1.0 at the output
            var gradientTensor = new AiDotNet.LinearAlgebra.Tensor<T>(outputShape);
            
            // Fill the tensor with the loss gradient value
            // This assumes the loss is a scalar applied to all outputs
            for (int i = 0; i < gradientTensor.Length; i++)
            {
                gradientTensor[i] = lossGradient;
            }
            
            // Use the existing Backpropagate method
            network.Backpropagate(gradientTensor);
        }
        
        /// <summary>
        /// Performs backward propagation for a Transformer network.
        /// </summary>
        /// <typeparam name="T">The numeric type used by the network.</typeparam>
        /// <param name="transformer">The transformer network.</param>
        /// <param name="lossGradient">The gradient of the loss with respect to the network output.</param>
        public static void Backward<T>(this Transformer<T> transformer, T lossGradient)
        {
            // Get the output shape from the architecture
            var outputShape = transformer.Architecture.GetOutputShape();
            
            // Create a tensor from the scalar loss gradient
            var gradientTensor = new AiDotNet.LinearAlgebra.Tensor<T>(outputShape);
            
            // Fill the tensor with the loss gradient value
            for (int i = 0; i < gradientTensor.Length; i++)
            {
                gradientTensor[i] = lossGradient;
            }
            
            // Use the existing Backpropagate method
            transformer.Backpropagate(gradientTensor);
        }
    }
    
    /// <summary>
    /// Extension methods for replay buffer operations.
    /// </summary>
    public static class ReplayBufferExtensions
    {
        /// <summary>
        /// Samples a batch of experiences from the buffer.
        /// </summary>
        /// <typeparam name="TState">The type used to represent the environment state.</typeparam>
        /// <typeparam name="TAction">The type used to represent actions.</typeparam>
        /// <typeparam name="T">The numeric type used for calculations.</typeparam>
        /// <param name="buffer">The replay buffer.</param>
        /// <param name="batchSize">The number of experiences to sample.</param>
        /// <returns>A tuple containing the sampled experiences.</returns>
        public static (TState[] states, TAction[] actions, T[] rewards, TState[] nextStates, bool[] dones) 
            Sample<TState, TAction, T>(this IReplayBuffer<TState, TAction, T> buffer, int batchSize)
        {
            // Use the existing SampleBatch method and convert to tuple format
            var batch = buffer.SampleBatch(batchSize);
            
            // Return the components as a tuple
            return (batch.States, batch.Actions, batch.Rewards, batch.NextStates, batch.Dones);
        }
    }
}