using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Factories;
using AiDotNet.Models.Options;
using AiDotNet.Enums;

namespace AiDotNet.ReinforcementLearning.Agents;

/// <summary>
/// Value network that predicts Q-values for MBPO.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class ValueNetwork<T>
{
    private readonly int _stateSize;
    private readonly int _actionSize;
    private readonly bool _continuous;
    private readonly INumericOperations<T> _numOps = default!;
    private readonly int _inputSize;
    
    // Neural network components
    private readonly FullyConnectedLayer<T>[] _hiddenLayers;
    private readonly FullyConnectedLayer<T> _outputLayer = default!;
    
    // Optimizer for training
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer = default!;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="ValueNetwork{T}"/> class.
    /// </summary>
    /// <param name="stateSize">The size of the state space.</param>
    /// <param name="actionSize">The size of the action space.</param>
    /// <param name="hiddenSizes">The sizes of the hidden layers.</param>
    /// <param name="learningRate">The learning rate for training.</param>
    /// <param name="continuous">Whether the action space is continuous.</param>
    public ValueNetwork(int stateSize, int actionSize, int[] hiddenSizes, double learningRate, bool continuous)
    {
        _stateSize = stateSize;
        _actionSize = actionSize;
        _continuous = continuous;
        _numOps = MathHelper.GetNumericOperations<T>();
        
        // Create optimizer
        var options = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            LearningRate = learningRate
        };
        _optimizer = OptimizerFactory<T, Tensor<T>, Tensor<T>>.CreateOptimizer(OptimizerType.Adam, options);
        
        // Define the input size (state + action)
        _inputSize = _stateSize + (_continuous ? _actionSize : _actionSize);
        
        // Create hidden layers
        _hiddenLayers = new FullyConnectedLayer<T>[hiddenSizes.Length];
        for (int i = 0; i < hiddenSizes.Length; i++)
        {
            _hiddenLayers[i] = new FullyConnectedLayer<T>(
                i == 0 ? _inputSize : hiddenSizes[i - 1],
                hiddenSizes[i],
                (IActivationFunction<T>?)null);
        }
        
        // Create output layer (single Q-value)
        _outputLayer = new FullyConnectedLayer<T>(
            hiddenSizes[hiddenSizes.Length - 1],
            1,
            (IActivationFunction<T>?)null);
    }
    
    /// <summary>
    /// Gets the Q-value for the given state-action pair.
    /// </summary>
    /// <param name="state">The current state.</param>
    /// <param name="action">The action to evaluate.</param>
    /// <returns>The Q-value.</returns>
    public T GetValue(Tensor<T> state, Vector<T> action)
    {
        // Convert action to tensor and process based on action type
        Tensor<T> actionTensor;
        
        if (_continuous)
        {
            actionTensor = Tensor<T>.FromVector(action);
        }
        else
        {
            // For discrete actions, we use the one-hot encoded action
            actionTensor = Tensor<T>.FromVector(action);
        }
        
        // Concatenate state and action
        Tensor<T> input = ConcatenateTensors(state, actionTensor);
        
        // Forward pass through network
        Tensor<T> hidden = input;
        foreach (var layer in _hiddenLayers)
        {
            hidden = layer.Forward(hidden);
        }
        
        // Get Q-value
        Tensor<T> qTensor = _outputLayer.Forward(hidden);
        
        // Extract scalar value
        return qTensor.Rank > 0 ? qTensor[0] : qTensor[0, 0];
    }
    
    /// <summary>
    /// Updates the value network.
    /// </summary>
    /// <param name="states">Batch of states.</param>
    /// <param name="actions">Batch of actions.</param>
    /// <param name="targetValues">Batch of target Q-values.</param>
    /// <returns>The value loss.</returns>
    public T Update(Tensor<T>[] states, Vector<T>[] actions, T[] targetValues)
    {
        int batchSize = states.Length;
        T totalLoss = _numOps.Zero;
        
        for (int i = 0; i < batchSize; i++)
        {
            // Get current Q-value
            T currentValue = GetValue(states[i], actions[i]);
            
            // Calculate MSE loss
            T diff = _numOps.Subtract(currentValue, targetValues[i]);
            T squaredError = _numOps.Multiply(diff, diff);
            
            totalLoss = _numOps.Add(totalLoss, squaredError);
        }
        
        // Calculate average loss
        totalLoss = _numOps.Divide(totalLoss, _numOps.FromDouble(batchSize));
        
        // Perform backward pass and update parameters
        // We need to compute gradients for each sample
        for (int i = 0; i < batchSize; i++)
        {
            // Forward pass to get the output
            var input = CombineStateAction(states[i], actions[i]);
            var output = input;
            
            // Pass through hidden layers
            foreach (var layer in _hiddenLayers)
            {
                output = layer.Forward(output);
            }
            
            // Pass through output layer
            output = _outputLayer.Forward(output);
            
            // Compute gradient of loss with respect to output
            // For MSE loss: gradient = 2 * (prediction - target)
            T prediction = output[0];
            T diff = _numOps.Subtract(prediction, targetValues[i]);
            T gradient = _numOps.Multiply(_numOps.FromDouble(2.0), diff);
            
            // Create gradient tensor
            var outputGradient = new Tensor<T>(new[] { 1 });
            outputGradient[0] = gradient;
            
            // Backward pass through layers
            var currentGradient = _outputLayer.Backward(outputGradient);
            
            for (int j = _hiddenLayers.Length - 1; j >= 0; j--)
            {
                currentGradient = _hiddenLayers[j].Backward(currentGradient);
            }
        }
        
        // Update parameters using the learning rate from optimizer options
        var learningRate = (_optimizer.GetOptions() as AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>)?.LearningRate ?? 0.001;
        T lr = _numOps.FromDouble(learningRate);
        
        // Update each layer's parameters
        foreach (var layer in _hiddenLayers)
        {
            layer.UpdateParameters(lr);
        }
        _outputLayer.UpdateParameters(lr);
        
        return totalLoss;
    }
    
    /// <summary>
    /// Gets all parameters of the value network as a single vector.
    /// </summary>
    /// <returns>A vector containing all network parameters.</returns>
    public Vector<T> GetParameters()
    {
        var allParameters = new List<Vector<T>>();
        
        // Add hidden layer parameters
        foreach (var layer in _hiddenLayers)
        {
            allParameters.Add(layer.GetParameters());
        }
        
        // Add output layer parameters
        allParameters.Add(_outputLayer.GetParameters());
        
        // Combine all parameters
        return ConcatenateVectors(allParameters);
    }
    
    /// <summary>
    /// Sets all parameters of the value network from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    public void SetParameters(Vector<T> parameters)
    {
        int index = 0;
        
        // Set hidden layer parameters
        foreach (var layer in _hiddenLayers)
        {
            int layerParamSize = layer.GetParameters().Length;
            var layerParams = ExtractVector(parameters, index, layerParamSize);
            layer.SetParameters(layerParams);
            index += layerParamSize;
        }
        
        // Set output layer parameters
        int outputParamSize = _outputLayer.GetParameters().Length;
        var outputParams = ExtractVector(parameters, index, outputParamSize);
        _outputLayer.SetParameters(outputParams);
    }
    
    // Helper methods
    
    /// <summary>
    /// Combines state and action into a single tensor.
    /// </summary>
    private Tensor<T> CombineStateAction(Tensor<T> state, Vector<T> action)
    {
        // Convert action vector to tensor
        var actionTensor = new Tensor<T>(new int[] { action.Length });
        for (int i = 0; i < action.Length; i++)
        {
            actionTensor[i] = action[i];
        }
        
        return ConcatenateTensors(state, actionTensor);
    }
    
    /// <summary>
    /// Concatenates two tensors.
    /// </summary>
    private Tensor<T> ConcatenateTensors(Tensor<T> a, Tensor<T> b)
    {
        // Simple implementation for 1D or flat tensors
        int aSize = a.Shape[0];
        int bSize = b.Shape[0];
        
        var result = new Tensor<T>(new int[] { aSize + bSize });
        
        for (int i = 0; i < aSize; i++)
        {
            var aIndex = a.GetIndexFromFlat(i);
            result[i] = a[aIndex];
        }
        
        for (int i = 0; i < bSize; i++)
        {
            var bIndex = b.GetIndexFromFlat(i);
            result[aSize + i] = b[bIndex];
        }
        
        return result;
    }
    
    /// <summary>
    /// Concatenates a list of vectors into a single vector.
    /// </summary>
    private Vector<T> ConcatenateVectors(List<Vector<T>> vectors)
    {
        // Calculate total length
        int totalLength = 0;
        foreach (var vector in vectors)
        {
            totalLength += vector.Length;
        }
        
        // Create result vector
        var result = new Vector<T>(totalLength);
        
        // Copy values
        int index = 0;
        foreach (var vector in vectors)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                result[index++] = vector[i];
            }
        }
        
        return result;
    }
    
    /// <summary>
    /// Extracts a portion of a vector.
    /// </summary>
    private Vector<T> ExtractVector(Vector<T> source, int startIndex, int length)
    {
        var result = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            result[i] = source[startIndex + i];
        }
        return result;
    }
    
    /// <summary>
    /// Saves the value network to a file.
    /// </summary>
    /// <param name="path">The file path to save to.</param>
    public void Save(string path)
    {
        var parameters = GetParameters();
        using (var writer = new System.IO.BinaryWriter(System.IO.File.Open(path, System.IO.FileMode.Create)))
        {
            writer.Write(_inputSize);
            writer.Write(_hiddenLayers.Length);
            foreach (var layer in _hiddenLayers)
            {
                writer.Write(layer.GetOutputShape()[0]);
            }
            writer.Write(parameters.Length);
            for (int i = 0; i < parameters.Length; i++)
            {
                writer.Write(Convert.ToDouble(parameters[i]));
            }
        }
    }
    
    /// <summary>
    /// Loads the value network from a file.
    /// </summary>
    /// <param name="path">The file path to load from.</param>
    public void Load(string path)
    {
        using (var reader = new System.IO.BinaryReader(System.IO.File.Open(path, System.IO.FileMode.Open)))
        {
            int inputSize = reader.ReadInt32();
            int numHiddenLayers = reader.ReadInt32();
            
            if (inputSize != _inputSize)
            {
                throw new InvalidOperationException("Network input size does not match");
            }
            
            // Skip hidden layer sizes - we already have them
            for (int i = 0; i < numHiddenLayers; i++)
            {
                reader.ReadInt32();
            }
            
            int paramCount = reader.ReadInt32();
            var parameters = new Vector<T>(paramCount);
            for (int i = 0; i < paramCount; i++)
            {
                parameters[i] = _numOps.FromDouble(reader.ReadDouble());
            }
            
            SetParameters(parameters);
        }
    }
}