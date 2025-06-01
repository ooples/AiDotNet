using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks
{
    /// <summary>
    /// Helper classes for Vision Transformer
    /// </summary>
    
    /// <summary>
    /// Layer normalization for Vision Transformer
    /// </summary>
    public class LayerNorm : ILayer
    {
        private readonly int features;
        private readonly double eps;
        private Tensor<double> gamma;
        private Tensor<double> beta;
        
        public LayerNorm(int features, double eps = 1e-5)
        {
            this.features = features;
            this.eps = eps;
            
            // Initialize learnable parameters
            gamma = new Tensor<double>(new[] { features });
            beta = new Tensor<double>(new[] { features });
            
            // Initialize gamma to 1 and beta to 0
            for (int i = 0; i < features; i++)
            {
                gamma.Data[i] = 1.0;
                beta.Data[i] = 0.0;
            }
        }
        
        public Tensor<double> Forward(Tensor<double> input)
        {
            // Compute mean and variance along last dimension
            var lastDim = input.Shape[input.Shape.Length - 1];
            var numElements = input.Data.Length / lastDim;
            
            var output = new Tensor<double>(input.Shape);
            
            for (int i = 0; i < numElements; i++)
            {
                // Compute mean
                double mean = 0;
                for (int j = 0; j < lastDim; j++)
                {
                    mean += input.Data[i * lastDim + j];
                }
                mean /= lastDim;
                
                // Compute variance
                double variance = 0;
                for (int j = 0; j < lastDim; j++)
                {
                    double diff = input.Data[i * lastDim + j] - mean;
                    variance += diff * diff;
                }
                variance /= lastDim;
                
                // Normalize and scale
                double std = Math.Sqrt(variance + eps);
                for (int j = 0; j < lastDim; j++)
                {
                    double normalized = (input.Data[i * lastDim + j] - mean) / std;
                    output.Data[i * lastDim + j] = gamma.Data[j] * normalized + beta.Data[j];
                }
            }
            
            return output;
        }
        
        public Tensor<double> Backward(Tensor<double> gradOutput)
        {
            // Simplified backward pass
            return gradOutput;
        }
        
        public string Name => "LayerNorm";
        public LayerType LayerType => LayerType.BatchNormalization;
        public int InputSize => features;
        public int OutputSize => features;
        public List<Tensor<double>> Parameters => new List<Tensor<double>> { gamma, beta };
        public List<Tensor<double>> Gradients => new List<Tensor<double>>();
    }
    
    /// <summary>
    /// Dense (fully connected) layer for Vision Transformer
    /// </summary>
    public class Dense : ILayer
    {
        private readonly int inputSize;
        private readonly int outputSize;
        private Tensor<double> weight;
        private Tensor<double> bias;
        
        public Dense(int inputSize, int outputSize)
        {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            
            weight = new Tensor<double>(new[] { inputSize, outputSize });
            bias = new Tensor<double>(new[] { outputSize });
            
            InitializeWeights();
        }
        
        public Tensor<double> Forward(Tensor<double> input)
        {
            // Simplified forward pass - assumes 2D or 3D input
            if (input.Shape.Length == 2)
            {
                return input.MatMul(weight).Add(bias);
            }
            else if (input.Shape.Length == 3)
            {
                // Batch matrix multiplication for 3D tensors
                var batchSize = input.Shape[0];
                var seqLen = input.Shape[1];
                var output = new Tensor<double>(new[] { batchSize, seqLen, outputSize });
                
                // Process each batch
                for (int b = 0; b < batchSize; b++)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        // Compute dot product for each position
                        for (int o = 0; o < outputSize; o++)
                        {
                            double sum = bias.Data[o];
                            for (int i = 0; i < inputSize; i++)
                            {
                                sum += input.Data[b * seqLen * inputSize + s * inputSize + i] * 
                                       weight.Data[i * outputSize + o];
                            }
                            output.Data[b * seqLen * outputSize + s * outputSize + o] = sum;
                        }
                    }
                }
                
                return output;
            }
            
            throw new NotSupportedException("Dense layer only supports 2D or 3D tensors");
        }
        
        public Tensor<double> Backward(Tensor<double> gradOutput)
        {
            // Simplified backward pass
            return gradOutput;
        }
        
        private void InitializeWeights()
        {
            // Xavier initialization
            var scale = Math.Sqrt(2.0 / (inputSize + outputSize));
            var random = new Random();
            
            for (int i = 0; i < weight.Data.Length; i++)
            {
                weight.Data[i] = (random.NextDouble() * 2 - 1) * scale;
            }
            
            Array.Clear(bias.Data, 0, bias.Data.Length);
        }
        
        public string Name => "Dense";
        public LayerType LayerType => LayerType.Dense;
        public int InputSize => inputSize;
        public int OutputSize => outputSize;
        public List<Tensor<double>> Parameters => new List<Tensor<double>> { weight, bias };
        public List<Tensor<double>> Gradients => new List<Tensor<double>>();
    }
    
    /// <summary>
    /// MLP (Multi-Layer Perceptron) for Vision Transformer
    /// </summary>
    public class MLP : ILayer
    {
        private readonly Dense fc1;
        private readonly Dense fc2;
        private readonly double dropoutRate;
        
        public MLP(int embedDim, int mlpDim, double dropoutRate)
        {
            this.dropoutRate = dropoutRate;
            fc1 = new Dense(embedDim, mlpDim);
            fc2 = new Dense(mlpDim, embedDim);
        }
        
        public Tensor<double> Forward(Tensor<double> input)
        {
            var x = fc1.Forward(input);
            x = ApplyGELU(x);
            x = Dropout(x, dropoutRate);
            x = fc2.Forward(x);
            x = Dropout(x, dropoutRate);
            return x;
        }
        
        public Tensor<double> Backward(Tensor<double> gradOutput)
        {
            // Simplified backward pass
            return gradOutput;
        }
        
        private Tensor<double> ApplyGELU(Tensor<double> input)
        {
            // GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            var output = new Tensor<double>(input.Shape);
            var sqrtTwoOverPi = Math.Sqrt(2.0 / Math.PI);
            
            for (int i = 0; i < input.Data.Length; i++)
            {
                var x = input.Data[i];
                var cube = x * x * x;
                var inner = sqrtTwoOverPi * (x + 0.044715 * cube);
                output.Data[i] = 0.5 * x * (1.0 + Math.Tanh(inner));
            }
            
            return output;
        }
        
        private Tensor<double> Dropout(Tensor<double> input, double rate)
        {
            // Simplified - no dropout during inference
            return input;
        }
        
        public string Name => "MLP";
        public LayerType LayerType => LayerType.Dense;
        public int InputSize => fc1.InputSize;
        public int OutputSize => fc2.OutputSize;
        public List<Tensor<double>> Parameters => new List<Tensor<double>>();
        public List<Tensor<double>> Gradients => new List<Tensor<double>>();
    }
}