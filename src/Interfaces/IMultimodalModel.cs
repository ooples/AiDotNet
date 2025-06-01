using AiDotNet.LinearAlgebra;
using System.Collections.Generic;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Interface for multimodal AI models that can process multiple input modalities
    /// </summary>
    public interface IMultimodalModel
    {
        /// <summary>
        /// Gets the supported modalities
        /// </summary>
        IReadOnlyList<string> SupportedModalities { get; }

        /// <summary>
        /// Gets the fusion strategy used by the model
        /// </summary>
        string FusionStrategy { get; }

        /// <summary>
        /// Processes multimodal input data
        /// </summary>
        /// <param name="modalityData">Dictionary mapping modality names to their data</param>
        /// <returns>Fused representation</returns>
        Vector<double> ProcessMultimodal(Dictionary<string, object> modalityData);

        /// <summary>
        /// Adds a modality encoder to the model
        /// </summary>
        /// <param name="modalityName">Name of the modality</param>
        /// <param name="encoder">The encoder for this modality</param>
        void AddModalityEncoder(string modalityName, IModalityEncoder encoder);

        /// <summary>
        /// Gets the encoder for a specific modality
        /// </summary>
        /// <param name="modalityName">Name of the modality</param>
        /// <returns>The modality encoder</returns>
        IModalityEncoder GetModalityEncoder(string modalityName);

        /// <summary>
        /// Sets the attention weights between modalities (for cross-attention models)
        /// </summary>
        /// <param name="weights">Matrix<double> of attention weights</param>
        void SetCrossModalityAttention(Matrix<double> weights);
    }

    /// <summary>
    /// Interface for modality-specific encoders
    /// </summary>
    public interface IModalityEncoder
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
        Vector<double> Encode(object input);

        /// <summary>
        /// Preprocesses the input data for this modality
        /// </summary>
        /// <param name="input">Raw input data</param>
        /// <returns>Preprocessed data ready for encoding</returns>
        object Preprocess(object input);
    }
}