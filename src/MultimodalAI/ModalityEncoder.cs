using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using System;

namespace AiDotNet.MultimodalAI
{
    /// <summary>
    /// Base class for modality-specific encoders
    /// </summary>
    public abstract class ModalityEncoder : IModalityEncoder
    {
        protected readonly int _outputDimension;
        protected readonly string _modalityName;
        protected NeuralNetwork<double> _encoder = default!;

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
        protected ModalityEncoder(string modalityName, int outputDimension)
        {
            if (string.IsNullOrWhiteSpace(modalityName))
                throw new ArgumentException("Modality name cannot be null or empty", nameof(modalityName));

            if (outputDimension <= 0)
                throw new ArgumentException("Output dimension must be positive", nameof(outputDimension));

            _modalityName = modalityName;
            _outputDimension = outputDimension;
        }

        /// <summary>
        /// Encodes the modality-specific input into a vector representation
        /// </summary>
        /// <param name="input">The input data for this modality</param>
        /// <returns>Encoded vector representation</returns>
        public abstract Vector<double> Encode(object input);

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
        protected virtual Vector<double> Normalize(Vector<double> encoded)
        {
            var magnitude = encoded.Magnitude();
            if (magnitude > 0)
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
        /// <returns>Vector<double> with dropout applied</returns>
        protected virtual Vector<double> ApplyDropout(Vector<double> encoded, double dropoutRate, Random random)
        {
            if (dropoutRate <= 0 || dropoutRate >= 1)
                return encoded;

            var result = new Vector<double>(encoded.Dimension);
            for (int i = 0; i < encoded.Dimension; i++)
            {
                if (random.NextDouble() > dropoutRate)
                {
                    result[i] = encoded[i] / (1 - dropoutRate);
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
        protected virtual Vector<double> Project(Vector<double> encoded, Matrix<double> projectionMatrix)
        {
            if (projectionMatrix == null)
                return encoded;

            if (encoded.Dimension != projectionMatrix.Columns)
                throw new ArgumentException($"Vector<double> dimension {encoded.Dimension} does not match projection matrix columns {projectionMatrix.Columns}");

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
    }
}