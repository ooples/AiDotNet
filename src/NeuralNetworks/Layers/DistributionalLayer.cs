using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers
{
    /// <summary>
    /// A neural network layer that outputs a categorical distribution over a specified number of atoms.
    /// This layer is used in distributional reinforcement learning algorithms like C51.
    /// </summary>
    /// <typeparam name="T">The numeric type used for computations.</typeparam>
    public class DistributionalLayer<T> : LayerBase<T>
    {
        private readonly int _outputSize;
        private readonly int _atomCount;
        private readonly SoftmaxActivation<T> _softmax = default!;
        private Tensor<T> _weights = default!;
        private Tensor<T> _biases = default!;
        private Tensor<T> _weightsGradient = default!;
        private Tensor<T> _biasesGradient = default!;
        private Tensor<T>? _input;
        private Tensor<T>? _output;
        private Tensor<T>? _logits;
        
        /// <summary>
        /// Gets the numeric operations for type T.
        /// </summary>
        // Using base NumOps property from LayerBase<T>

        /// <summary>
        /// Gets a value indicating whether this layer supports training.
        /// </summary>
        public override bool SupportsTraining => true;

        /// <summary>
        /// Initializes a new instance of the <see cref="DistributionalLayer{T}"/> class.
        /// </summary>
        /// <param name="inputShape">Shape of the input data.</param>
        /// <param name="outputSize">Number of output units (usually the action space size).</param>
        /// <param name="atomCount">Number of atoms in the support distribution.</param>
        public DistributionalLayer(int[] inputShape, int outputSize, int atomCount)
            : base(inputShape, new[] { outputSize, atomCount })
        {
            _outputSize = outputSize;
            _atomCount = atomCount;
            _softmax = new SoftmaxActivation<T>();
            
            // Calculate the input size from the input shape
            int inputSize = inputShape[0];
            
            var scale = NumOps.FromDouble(Math.Sqrt(1.0 / inputSize));
            var minValue = NumOps.Multiply(NumOps.FromDouble(-1), scale);
            
            // Initialize weights and biases
            // Create a tensor with random values between -scale and scale
            _weights = new Tensor<T>(new[] { inputSize, _outputSize * _atomCount });
            
            // Fill with random values
            Random random = new Random();
            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < _outputSize * _atomCount; j++)
                {
                    double randomValue = random.NextDouble() * (2 * Convert.ToDouble(scale)) - Convert.ToDouble(scale);
                    _weights[i, j] = NumOps.FromDouble(randomValue);
                }
            }
            
            // Initialize biases to zeros
            _biases = new Tensor<T>(new[] { _outputSize * _atomCount });
            for (int i = 0; i < _outputSize * _atomCount; i++)
            {
                _biases[i] = NumOps.Zero;
            }
            
            // Initialize gradients with zeros
            _weightsGradient = new Tensor<T>(new[] { inputSize, _outputSize * _atomCount });
            _biasesGradient = new Tensor<T>(new[] { _outputSize * _atomCount });
        }

        /// <summary>
        /// Gets the number of atoms in the support distribution.
        /// </summary>
        public int AtomCount => _atomCount;

        /// <summary>
        /// Gets the number of output units (usually the action space size).
        /// </summary>
        public override int OutputSize => _outputSize;

        /// <summary>
        /// Performs the forward pass of the layer.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public override Tensor<T> Forward(Tensor<T> input)
        {
            // Save input for backward pass
            _input = input;

            // Compute logits: input * weights + biases
            var batchSize = input.Shape[0];
            var flatLogits = input.MatrixMultiply(_weights).Add(_biases);
            
            // Reshape to [batch_size, output_size, atom_count]
            _logits = flatLogits.Reshape(batchSize, _outputSize, _atomCount);
            
            // Apply softmax to get probabilities for each action
            var distributions = new Tensor<T>(new[] { batchSize, _outputSize, _atomCount });
            
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < _outputSize; j++)
                {
                    // Get tensor slice for softmax activation
                    var actionLogitsSlice = _logits.GetSlice(i, j);
                    // Convert tensor to vector for the softmax activation
                    var actionLogitsVector = new Vector<T>(actionLogitsSlice.Length);
                    for (int k = 0; k < actionLogitsSlice.Length; k++)
                    {
                        actionLogitsVector[k] = actionLogitsSlice[k];
                    }
                    var actionProbsVector = _softmax.Activate(actionLogitsVector);
                    
                    for (int k = 0; k < _atomCount; k++)
                    {
                        distributions[i, j, k] = actionProbsVector[k];
                    }
                }
            }
            
            _output = distributions;
            return _output;
        }

        /// <summary>
        /// Computes the gradient of softmax with respect to its inputs, combined with the output gradient.
        /// </summary>
        /// <param name="softmaxOutput">The output of the softmax function.</param>
        /// <param name="outputGradient">The gradient of the loss with respect to the softmax output.</param>
        /// <returns>The gradient of the loss with respect to the softmax input.</returns>
        private Vector<T> ComputeSoftmaxGradient(Vector<T> softmaxOutput, Vector<T> outputGradient)
        {
            // The Jacobian of softmax for each element i with respect to each input j is:
            // J_ij = softmax_i * (delta_ij - softmax_j)
            // where delta_ij is 1 if i=j and 0 otherwise.
            // We compute J * outputGradient directly without explicitly forming J.
            
            // First compute the dot product of softmax outputs and output gradients
            T dotProduct = NumOps.Zero;
            for (int i = 0; i < softmaxOutput.Length; i++)
            {
                dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(softmaxOutput[i], outputGradient[i]));
            }
            
            // Compute the gradient for each element: softmax_i * (output_gradient_i - dot_product)
            var gradient = new Vector<T>(softmaxOutput.Length);
            for (int i = 0; i < softmaxOutput.Length; i++)
            {
                T gradTerm = NumOps.Subtract(outputGradient[i], dotProduct);
                gradient[i] = NumOps.Multiply(softmaxOutput[i], gradTerm);
            }
            
            return gradient;
        }

        /// <summary>
        /// Performs the backward pass of the layer.
        /// </summary>
        /// <param name="outputGradient">The gradient of the loss with respect to the output.</param>
        /// <returns>The gradient of the loss with respect to the input.</returns>
        public override Tensor<T> Backward(Tensor<T> outputGradient)
        {
            if (_input == null || _output == null || _logits == null)
            {
                throw new InvalidOperationException("Forward pass must be called before backward pass.");
            }

            var batchSize = _input.Shape[0];
            var inputSize = _input.Shape[1];
            
            // Initialize gradients with zeros
            // Use nested loops to fill tensors with zeros
            for (int i = 0; i < _weightsGradient.Shape[0]; i++)
            {
                for (int j = 0; j < _weightsGradient.Shape[1]; j++)
                {
                    _weightsGradient[i, j] = NumOps.Zero;
                }
            }
            
            for (int i = 0; i < _biasesGradient.Shape[0]; i++)
            {
                _biasesGradient[i] = NumOps.Zero;
            }
            
            // Compute softmax gradients for each action
            var flatGradient = new Tensor<T>(new[] { batchSize, _outputSize * _atomCount });
            
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < _outputSize; j++)
                {
                    // Get the distributions (softmax outputs) and output gradients for this action
                    var distsSlice = _output.GetSlice(i, j);
                    var gradsSlice = outputGradient.GetSlice(i, j);
                    
                    // Convert tensor slices to vectors
                    var distsVector = new Vector<T>(distsSlice.Length);
                    var gradsVector = new Vector<T>(gradsSlice.Length);
                    
                    for (int k = 0; k < distsSlice.Length; k++)
                    {
                        distsVector[k] = distsSlice[k];
                        gradsVector[k] = gradsSlice[k];
                    }
                    
                    // Compute softmax gradients
                    var softmaxGradsVector = ComputeSoftmaxGradient(distsVector, gradsVector);
                    
                    // Copy the gradients to the flat gradient tensor
                    for (int k = 0; k < _atomCount; k++)
                    {
                        flatGradient[i, j * _atomCount + k] = softmaxGradsVector[k];
                    }
                }
            }
            
            // Compute weight and bias gradients
            _weightsGradient = _input.Transpose().MatrixMultiply(flatGradient);
            
            // Compute sum along the batch dimension for bias gradients
            _biasesGradient = new Tensor<T>(new[] { _outputSize * _atomCount });
            for (int i = 0; i < _outputSize * _atomCount; i++)
            {
                T sum = NumOps.Zero;
                for (int b = 0; b < batchSize; b++)
                {
                    sum = NumOps.Add(sum, flatGradient[b, i]);
                }
                _biasesGradient[i] = sum;
            }
            
            // Compute input gradient: gradient * weights^T
            var inputGradient = flatGradient.MatrixMultiply(_weights.Transpose());
            
            return inputGradient;
        }

        /// <summary>
        /// Updates the layer parameters using the calculated gradients.
        /// </summary>
        /// <param name="learningRate">The learning rate to use for the update.</param>
        public override void UpdateParameters(T learningRate)
        {
            // Update weights and biases using gradients
            _weights = _weights.Subtract(_weightsGradient.Multiply(learningRate));
            _biases = _biases.Subtract(_biasesGradient.Multiply(learningRate));
        }

        /// <summary>
        /// Gets the parameters of the layer.
        /// </summary>
        /// <returns>A vector containing the layer's parameters.</returns>
        public override Vector<T> GetParameters()
        {
            // Calculate total parameter count
            int weightCount = _weights.Shape[0] * _weights.Shape[1];
            int biasCount = _biases.Shape[0];
            Vector<T> parameters = new Vector<T>(weightCount + biasCount);
            
            // Copy weights to parameters
            int index = 0;
            for (int i = 0; i < _weights.Shape[0]; i++)
            {
                for (int j = 0; j < _weights.Shape[1]; j++)
                {
                    parameters[index++] = _weights[i, j];
                }
            }
            
            // Copy biases to parameters
            for (int i = 0; i < _biases.Shape[0]; i++)
            {
                parameters[index++] = _biases[i];
            }
            
            return parameters;
        }

        /// <summary>
        /// Sets the parameters of the layer.
        /// </summary>
        /// <param name="parameters">A vector containing the layer's parameters.</param>
        public override void SetParameters(Vector<T> parameters)
        {
            int weightCount = _weights.Shape[0] * _weights.Shape[1];
            int biasCount = _biases.Shape[0];
            
            if (parameters.Length != weightCount + biasCount)
            {
                throw new ArgumentException($"Expected {weightCount + biasCount} parameters, but got {parameters.Length}");
            }
            
            // Copy parameters to weights
            int index = 0;
            for (int i = 0; i < _weights.Shape[0]; i++)
            {
                for (int j = 0; j < _weights.Shape[1]; j++)
                {
                    _weights[i, j] = parameters[index++];
                }
            }
            
            // Copy parameters to biases
            for (int i = 0; i < _biases.Shape[0]; i++)
            {
                _biases[i] = parameters[index++];
            }
        }

        /// <summary>
        /// Resets the internal state of the layer.
        /// </summary>
        public override void ResetState()
        {
            _input = null;
            _output = null;
            _logits = null;
        }
    }
}