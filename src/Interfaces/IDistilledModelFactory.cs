namespace AiDotNet.Interfaces;

using AiDotNet.Interfaces;
using AiDotNet.Compression.KnowledgeDistillation;
using System.IO;

/// <summary>
/// Interface for factories that can create distilled models.
/// </summary>
/// <remarks>
/// <para>
/// This interface defines methods for creating distilled models from serialized data.
/// </para>
/// <para><b>For Beginners:</b> This defines how to create distilled models from saved data.
/// 
/// A factory implementing this interface knows:
/// - How to read the serialized data for a specific model type
/// - How to reconstruct the distilled model from that data
/// - How to set up the model with its distillation metadata
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TModel">The type of model.</typeparam>
/// <typeparam name="TInput">The input type for the model.</typeparam>
/// <typeparam name="TOutput">The output type for the model.</typeparam>
public interface IDistilledModelFactory<T, TModel, TInput, TOutput>
    where T : unmanaged
    where TModel : class, IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Deserializes a distilled model from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader containing the serialized model.</param>
    /// <param name="method">The distillation method used.</param>
    /// <param name="useSoftTargets">Whether soft targets were used for distillation.</param>
    /// <param name="temperature">The temperature used for distillation.</param>
    /// <param name="compressionRatio">The achieved compression ratio.</param>
    /// <returns>The deserialized distilled model.</returns>
    /// <remarks>
    /// <para>
    /// This method reads serialized data and constructs a distilled model from it.
    /// </para>
    /// <para><b>For Beginners:</b> This rebuilds a distilled model from saved data.
    /// 
    /// The factory reads the serialized data and:
    /// 1. Creates the appropriate student model instance
    /// 2. Sets up the model parameters
    /// 3. Configures the distillation metadata
    /// 4. Returns a fully functional distilled model
    /// </para>
    /// </remarks>
    TModel DeserializeDistilledModel(
        BinaryReader reader,
        DistillationMethod method,
        bool useSoftTargets,
        double temperature,
        double compressionRatio);
}