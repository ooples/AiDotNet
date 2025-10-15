namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for multimodal AI models that can process multiple input modalities
/// </summary>
/// <typeparam name="T">The numeric type used for calculations</typeparam>
public interface IMultimodalModel<T> : IFullModel<T, Dictionary<string, object>, Vector<T>>
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
    Vector<T> ProcessMultimodal(Dictionary<string, object> modalityData);

    /// <summary>
    /// Adds a modality encoder to the model
    /// </summary>
    /// <param name="modalityName">Name of the modality</param>
    /// <param name="encoder">The encoder for this modality</param>
    void AddModalityEncoder(string modalityName, IModalityEncoder<T> encoder);

    /// <summary>
    /// Gets the encoder for a specific modality
    /// </summary>
    /// <param name="modalityName">Name of the modality</param>
    /// <returns>The modality encoder</returns>
    IModalityEncoder<T> GetModalityEncoder(string modalityName);

    /// <summary>
    /// Sets the attention weights between modalities (for cross-attention models)
    /// </summary>
    /// <param name="weights">Matrix of attention weights</param>
    void SetCrossModalityAttention(Matrix<T> weights);
    
    /// <summary>
    /// Adds a new modality to the model
    /// </summary>
    /// <param name="modalityName">Name of the modality to add</param>
    /// <param name="dimension">Dimension of the modality</param>
    void AddModality(string modalityName, int dimension);
    
    /// <summary>
    /// Sets the fusion strategy for combining modalities
    /// </summary>
    /// <param name="strategy">The fusion strategy to use</param>
    void SetFusionStrategy(string strategy);
}