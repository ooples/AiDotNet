namespace AiDotNet.Interfaces;

using AiDotNet.Interfaces;
using System.Collections.Generic;
using System.IO;

/// <summary>
/// Interface for models that are the result of knowledge distillation.
/// </summary>
/// <remarks>
/// <para>
/// This interface should be implemented by models that have been trained
/// through knowledge distillation.
/// </para>
/// <para><b>For Beginners:</b> This marks a model as being a distilled student model.
/// 
/// A distilled model:
/// - Was trained by mimicking a larger teacher model
/// - Contains metadata about the distillation process
/// - Typically has far fewer parameters than the original model
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input type for the model.</typeparam>
/// <typeparam name="TOutput">The output type for the model.</typeparam>
public interface IDistilledModel<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>
    where T : unmanaged
{
    /// <summary>
    /// Gets the temperature used during distillation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The temperature parameter controls how "soft" the teacher model's probability
    /// distribution was during distillation.
    /// </para>
    /// <para><b>For Beginners:</b> This is the temperature value used during training.
    /// 
    /// Higher temperature values:
    /// - Make probability distributions more uniform
    /// - Help the student learn more from less confident predictions
    /// - Typically range from 1 to 10
    /// </para>
    /// </remarks>
    double DistillationTemperature { get; }

    /// <summary>
    /// Gets the compression ratio achieved through distillation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The compression ratio is the ratio of the teacher model size to the student model size.
    /// </para>
    /// <para><b>For Beginners:</b> This tells how much smaller the model is than the original.
    /// 
    /// For example:
    /// - A value of 4.0 means the student is 1/4 the size of the teacher
    /// - Higher values indicate greater compression
    /// </para>
    /// </remarks>
    double CompressionRatio { get; }

    /// <summary>
    /// Gets structural metrics about the distilled model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This dictionary contains metrics about the structural differences between
    /// the teacher and student models.
    /// </para>
    /// <para><b>For Beginners:</b> This shows how the student model structure differs from the teacher.
    /// 
    /// For example:
    /// - Number of layers in teacher vs. student
    /// - Size of each layer in teacher vs. student
    /// - Other architectural differences
    /// </para>
    /// </remarks>
    IDictionary<string, object> StructuralMetrics { get; }

    /// <summary>
    /// Serializes the distilled model to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the distilled model's data to a binary stream.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the distilled model to a stream.
    /// 
    /// The method writes:
    /// - The model parameters
    /// - Distillation metadata
    /// - Any model-specific configuration
    /// </para>
    /// </remarks>
    void SerializeDistilled(BinaryWriter writer);
}