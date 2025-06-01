using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Non-generic interface for neural network layers
    /// </summary>
    public interface ILayer
    {
        /// <summary>
        /// Gets the name of the layer
        /// </summary>
        string Name { get; }
        
        /// <summary>
        /// Gets the type of layer
        /// </summary>
        LayerType LayerType { get; }
        
        /// <summary>
        /// Gets the input size of the layer
        /// </summary>
        int InputSize { get; }
        
        /// <summary>
        /// Gets the output size of the layer
        /// </summary>
        int OutputSize { get; }
        
        /// <summary>
        /// Gets the parameters of the layer
        /// </summary>
        List<Tensor<double>> Parameters { get; }
        
        /// <summary>
        /// Gets the gradients for the layer parameters
        /// </summary>
        List<Tensor<double>> Gradients { get; }
        
        /// <summary>
        /// Forward pass through the layer
        /// </summary>
        Tensor<double> Forward(Tensor<double> input);
        
        /// <summary>
        /// Backward pass through the layer
        /// </summary>
        Tensor<double> Backward(Tensor<double> gradOutput);
    }
}