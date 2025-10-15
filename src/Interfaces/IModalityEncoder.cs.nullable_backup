using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Interface for modality-specific encoders
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public interface IModalityEncoder<T>
    {
        /// <summary>
        /// Gets the name of the modality this encoder handles
        /// </summary>
        string ModalityName { get; }

        /// <summary>
        /// Gets the output dimension of the encoded representation
        /// </summary>
        int OutputDimension { get; }

        /// <summary>
        /// Encodes the modality-specific input into a vector representation
        /// </summary>
        /// <param name="input">The input data for this modality</param>
        /// <returns>Encoded vector representation</returns>
        Vector<T> Encode(object input);

        /// <summary>
        /// Preprocesses the input data for this modality
        /// </summary>
        /// <param name="input">Raw input data</param>
        /// <returns>Preprocessed data ready for encoding</returns>
        object Preprocess(object input);
    }
}