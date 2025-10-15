using System;
using AiDotNet.Helpers;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.Deployment.Techniques
{
    /// <summary>
    /// Quantized neural network model that uses reduced precision for weights and activations.
    /// </summary>
    /// <typeparam name="T">The numeric type for calculations.</typeparam>
    public class QuantizedNeuralNetwork<T> : NeuralNetworkBase<T>, INeuralNetworkModel<T>
        where T : struct, IComparable<T>, IConvertible, IEquatable<T>
    {
        private readonly NeuralNetworkArchitecture<T> _architecture;
        private readonly string _quantizationType;
        private readonly INumericOperations<T> _numOps;

        /// <summary>
        /// Initializes a new instance of the QuantizedNeuralNetwork class.
        /// </summary>
        /// <param name="architecture">The network architecture.</param>
        /// <param name="quantizationType">The type of quantization applied.</param>
        public QuantizedNeuralNetwork(NeuralNetworkArchitecture<T> architecture, string quantizationType) 
            : base(architecture, new MeanSquaredErrorLoss<T>(), 1.0)
        {
            _architecture = architecture ?? throw new ArgumentNullException(nameof(architecture));
            _quantizationType = quantizationType ?? throw new ArgumentNullException(nameof(quantizationType));
            _numOps = MathHelper.GetNumericOperations<T>();
        }

        /// <summary>
        /// Gets the network architecture.
        /// </summary>
        public override NeuralNetworkArchitecture<T> GetArchitecture()
        {
            return _architecture;
        }

        /// <summary>
        /// Gets the input shape of the network.
        /// </summary>
        public int[] GetInputShape()
        {
            // Return input shape based on first layer
            if (_architecture.Layers != null && _architecture.Layers.Count > 0)
            {
                return _architecture.Layers[0].GetInputShape();
            }
            return new[] { 1 };
        }

        /// <summary>
        /// Gets the layer activations for a given input.
        /// </summary>
        public override Dictionary<string, Tensor<T>> GetLayerActivations(Tensor<T> input)
        {
            var activations = new Dictionary<string, Tensor<T>>();
            var currentInput = input;

            if (_architecture.Layers != null)
            {
                int layerIndex = 0;
                foreach (var layer in _architecture.Layers)
                {
                    var output = layer.Forward(currentInput);
                    activations[$"layer_{layerIndex}"] = output;
                    currentInput = output;
                    layerIndex++;
                }
            }

            return activations;
        }

        /// <summary>
        /// Initializes the layers of the network.
        /// </summary>
        protected override void InitializeLayers()
        {
            // Layers are already initialized from the provided architecture
            Layers.Clear();
            if (_architecture.Layers != null)
            {
                foreach (var layer in _architecture.Layers)
                {
                    if (layer is ILayer<T> typedLayer)
                    {
                        Layers.Add(typedLayer);
                    }
                }
            }
        }

        /// <summary>
        /// Serializes network-specific data.
        /// </summary>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(_quantizationType);
            writer.Write(_architecture.Layers.Count);
        }

        /// <summary>
        /// Deserializes network-specific data.
        /// </summary>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            // Read quantization type and layer count (already handled in base deserialization)
        }

        /// <summary>
        /// Creates a new instance of the quantized neural network.
        /// </summary>
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new QuantizedNeuralNetwork<T>(_architecture, _quantizationType);
        }

        /// <summary>
        /// Makes a prediction using the quantized network.
        /// </summary>
        public override Tensor<T> Predict(Tensor<T> input)
        {
            var currentInput = input;
            
            foreach (var layer in Layers)
            {
                currentInput = layer.Forward(currentInput);
            }
            
            return currentInput;
        }

        /// <summary>
        /// Trains the quantized network (not supported).
        /// </summary>
        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            // Quantized networks typically don't support training
            throw new NotSupportedException("Quantized neural networks do not support training. Train the original model first, then quantize it.");
        }

        /// <summary>
        /// Gets metadata about the quantized network.
        /// </summary>
        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                ModelType = ModelType.NeuralNetwork,
                FeatureCount = 0,
                Complexity = Layers.Count,
                Description = $"Quantized Neural Network with {_quantizationType} quantization",
                AdditionalInfo = new Dictionary<string, object>
                {
                    ["QuantizationType"] = _quantizationType,
                    ["LayerCount"] = Layers.Count,
                    ["IsQuantized"] = true
                }
            };
        }

        /// <summary>
        /// Updates the network parameters with new values.
        /// </summary>
        /// <param name="parameters">The new parameter values.</param>
        public override void UpdateParameters(Vector<T> parameters)
        {
            if (parameters == null)
            {
                throw new ArgumentNullException(nameof(parameters));
            }

            // For quantized networks, we need to update the parameters
            // and maintain quantization
            var paramIndex = 0;
            
            foreach (var layer in Layers)
            {
                if (layer is QuantizedLayer<T> quantizedLayer)
                {
                    var layerParamCount = quantizedLayer.ParameterCount;
                    if (layerParamCount > 0)
                    {
                        var layerParams = new T[layerParamCount];
                        for (int i = 0; i < layerParamCount; i++)
                        {
                            if (paramIndex < parameters.Length)
                            {
                                layerParams[i] = parameters[paramIndex++];
                            }
                        }
                        
                        // Update the layer parameters while maintaining quantization
                        quantizedLayer.UpdateParameters(new Vector<T>(layerParams));
                    }
                }
            }
        }
    }
}