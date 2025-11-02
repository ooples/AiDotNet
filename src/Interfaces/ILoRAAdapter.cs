namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for LoRA (Low-Rank Adaptation) adapters that wrap existing layers with parameter-efficient adaptations.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// LoRA adapters enable efficient fine-tuning of neural networks by learning low-rank decompositions
/// of weight updates instead of modifying all weights directly. This interface defines the contract
/// for all LoRA adapter implementations across different layer types.
/// </para>
/// <para><b>For Beginners:</b> A LoRA adapter wraps an existing layer (like a dense or convolutional layer)
/// and adds a small "correction layer" that learns what adjustments are needed. This is much more
/// memory-efficient than retraining all the weights in a large model.
///
/// Think of it like:
/// - The base layer has the original knowledge (frozen or trainable)
/// - The LoRA layer learns a small correction
/// - The final output combines both: original + correction
///
/// This allows you to adapt large pre-trained models with 100x fewer trainable parameters!
/// </para>
/// </remarks>
public interface ILoRAAdapter<T> : ILayer<T>
{
    /// <summary>
    /// Gets the base layer being adapted with LoRA.
    /// </summary>
    /// <remarks>
    /// This is the original layer that's being enhanced with LoRA adaptations.
    /// It may be frozen (non-trainable) during fine-tuning for maximum efficiency.
    /// </remarks>
    ILayer<T> BaseLayer { get; }

    /// <summary>
    /// Gets the LoRA layer providing the low-rank adaptation.
    /// </summary>
    /// <remarks>
    /// This layer implements the low-rank decomposition (A and B matrices)
    /// that provides the adaptation to the base layer's behavior.
    /// </remarks>
    LoRALayer<T> LoRALayer { get; }

    /// <summary>
    /// Gets whether the base layer's parameters are frozen during training.
    /// </summary>
    /// <remarks>
    /// When true, only the LoRA parameters are trained, dramatically reducing
    /// memory requirements and training time. This is the typical use case for LoRA.
    /// </remarks>
    bool IsBaseLayerFrozen { get; }

    /// <summary>
    /// Gets the rank of the low-rank decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The rank determines how many parameters the LoRA adaptation uses.
    /// Lower rank = fewer parameters = more efficient but less flexible.
    /// </para>
    /// <para>
    /// Typical values:
    /// - rank=1-4: Very efficient, minimal parameters
    /// - rank=8: Good balance (default for many applications)
    /// - rank=16-32: More flexibility, more parameters
    /// - rank=64+: Diminishing returns, approaching full fine-tuning
    /// </para>
    /// </remarks>
    int Rank { get; }

    /// <summary>
    /// Gets the scaling factor (alpha) for the LoRA adaptation.
    /// </summary>
    /// <remarks>
    /// Alpha controls how strongly the LoRA adaptation affects the output.
    /// The actual LoRA contribution is scaled by alpha/rank.
    /// Common practice: alpha = rank (scaling factor of 1.0)
    /// </remarks>
    double Alpha { get; }

    /// <summary>
    /// Merges the LoRA weights back into the original layer for deployment.
    /// </summary>
    /// <returns>A new layer with the LoRA adaptation baked into the weights.</returns>
    /// <remarks>
    /// <para>
    /// After training, you can merge the LoRA weights into the base layer to create
    /// a single layer that includes the adaptations. This:
    /// - Removes the overhead of parallel computation
    /// - Makes inference as fast as the original layer
    /// - Allows deployment without the LoRA infrastructure
    /// </para>
    /// <para><b>For Beginners:</b> Think of this as "baking in" your corrections.
    /// During training, you have original + correction computed separately.
    /// After merging, you have a single updated layer that includes both,
    /// making it faster to use in production.
    /// </para>
    /// </remarks>
    ILayer<T> MergeToOriginalLayer();
}
