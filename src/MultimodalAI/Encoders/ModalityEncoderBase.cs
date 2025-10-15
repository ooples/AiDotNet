using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Enums;
using System;
using System.Collections.Generic;

namespace AiDotNet.MultimodalAI.Encoders
{
    /// <summary>
    /// Base class for modality-specific encoders
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public abstract class ModalityEncoderBase<T> : IModalityEncoder<T>, IDisposable
    {
        protected readonly int _outputDimension;
        protected readonly string _modalityName;
        protected readonly INumericOperations<T> _numericOps;
        protected readonly INeuralNetworkModel<T>? _encoder;
        private readonly bool _ownsEncoder;

        /// <summary>
        /// Gets the name of the modality this encoder handles
        /// </summary>
        public string ModalityName => _modalityName;

        /// <summary>
        /// Gets the output dimension of the encoded representation
        /// </summary>
        public int OutputDimension => _outputDimension;

        /// <summary>
        /// Initializes a new instance of ModalityEncoder
        /// </summary>
        /// <param name="modalityName">Name of the modality</param>
        /// <param name="outputDimension">Output dimension of the encoder</param>
        /// <param name="encoder">Optional custom neural network encoder. If null, a default encoder will be created when needed.</param>
        protected ModalityEncoderBase(string modalityName, int outputDimension, INeuralNetworkModel<T>? encoder = null)
        {
            if (string.IsNullOrWhiteSpace(modalityName))
                throw new ArgumentException("Modality name cannot be null or empty", nameof(modalityName));

            if (outputDimension <= 0)
                throw new ArgumentException("Output dimension must be positive", nameof(outputDimension));

            _modalityName = modalityName;
            _outputDimension = outputDimension;
            _numericOps = MathHelper.GetNumericOperations<T>();
            _encoder = encoder;
            _ownsEncoder = encoder == null; // We own the encoder if we create it
        }

        /// <summary>
        /// Encodes the modality-specific input into a vector representation
        /// </summary>
        /// <param name="input">The input data for this modality</param>
        /// <returns>Encoded vector representation</returns>
        public abstract Vector<T> Encode(object input);

        /// <summary>
        /// Preprocesses the input data for this modality
        /// </summary>
        /// <param name="input">Raw input data</param>
        /// <returns>Preprocessed data ready for encoding</returns>
        public abstract object Preprocess(object input);

        /// <summary>
        /// Normalizes the encoded vector
        /// </summary>
        /// <param name="encoded">The encoded vector</param>
        /// <returns>Normalized vector</returns>
        protected virtual Vector<T> Normalize(Vector<T> encoded)
        {
            var magnitude = encoded.Magnitude();
            if (_numericOps.GreaterThan(magnitude, _numericOps.Zero))
            {
                return encoded / magnitude;
            }
            return encoded;
        }

        /// <summary>
        /// Applies dropout to the encoded representation
        /// </summary>
        /// <param name="encoded">The encoded vector</param>
        /// <param name="dropoutRate">Dropout rate (0-1)</param>
        /// <param name="random">Random number generator</param>
        /// <returns>Vector with dropout applied</returns>
        protected virtual Vector<T> ApplyDropout(Vector<T> encoded, double dropoutRate, Random random)
        {
            if (dropoutRate <= 0 || dropoutRate >= 1)
                return encoded;

            var result = new Vector<T>(encoded.Length);
            for (int i = 0; i < encoded.Length; i++)
            {
                if (random.NextDouble() > dropoutRate)
                {
                    result[i] = _numericOps.Divide(encoded[i], _numericOps.FromDouble(1 - dropoutRate));
                }
            }
            return result;
        }

        /// <summary>
        /// Projects the encoded vector to the target dimension
        /// </summary>
        /// <param name="encoded">The encoded vector</param>
        /// <param name="projectionMatrix">Projection matrix</param>
        /// <returns>Projected vector</returns>
        protected virtual Vector<T> Project(Vector<T> encoded, Matrix<T> projectionMatrix)
        {
            if (projectionMatrix == null)
                return encoded;

            if (encoded.Length != projectionMatrix.Columns)
                throw new ArgumentException($"Vector dimension {encoded.Length} does not match projection matrix columns {projectionMatrix.Columns}");

            return projectionMatrix * encoded;
        }

        /// <summary>
        /// Validates the input for this modality
        /// </summary>
        /// <param name="input">The input to validate</param>
        /// <returns>True if the input is valid</returns>
        protected abstract bool ValidateInput(object input);

        /// <summary>
        /// Gets a string representation of the encoder
        /// </summary>
        /// <returns>String representation</returns>
        public override string ToString()
        {
            return $"{GetType().Name}(modality={_modalityName}, output_dim={_outputDimension})";
        }

        /// <summary>
        /// Creates a default encoder network for this modality
        /// </summary>
        /// <param name="inputDimension">Input dimension for the encoder</param>
        /// <returns>A default neural network encoder</returns>
        protected virtual INeuralNetworkModel<T> CreateDefaultEncoder(int inputDimension)
        {
            // Create layers manually for optimal modality encoding
            var layers = new List<ILayer<T>>
            {
                // Input processing layer - transforms input to intermediate size
                new DenseLayer<T>(inputDimension, 256, new ReLUActivation<T>()),
                
                // Hidden processing layer for feature extraction
                new DenseLayer<T>(256, 128, new ReLUActivation<T>()),
                
                // Output layer with normalized output (Tanh gives [-1,1] range)
                new DenseLayer<T>(128, _outputDimension, new TanhActivation<T>())
            };
            
            // Create architecture with the custom layers
            var architecture = new NeuralNetworkArchitecture<T>(
                complexity: NetworkComplexity.Medium,
                layers: layers
            );
            
            return new FeedForwardNeuralNetwork<T>(architecture);
        }

        /// <summary>
        /// Gets or creates the encoder network
        /// </summary>
        /// <param name="inputDimension">Input dimension for the encoder</param>
        /// <returns>The encoder network</returns>
        protected INeuralNetworkModel<T> GetOrCreateEncoder(int inputDimension)
        {
            if (_encoder != null)
                return _encoder;

            // Create default encoder - note this is not thread-safe
            // In production, you might want to use lazy initialization or locking
            return CreateDefaultEncoder(inputDimension);
        }

        /// <summary>
        /// Disposes of the encoder resources
        /// </summary>
        public virtual void Dispose()
        {
            // Only dispose the encoder if we created it
            if (_ownsEncoder && _encoder is IDisposable disposableEncoder)
            {
                disposableEncoder.Dispose();
            }
        }
    }
}