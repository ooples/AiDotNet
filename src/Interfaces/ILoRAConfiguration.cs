namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for configuring how LoRA (Low-Rank Adaptation) should be applied to neural network layers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This interface defines a strategy pattern for applying LoRA adaptations to layers within a model.
/// Different implementations can provide different strategies for which layers to adapt and how.
/// </para>
/// <para><b>For Beginners:</b> This interface lets you define a "strategy" for how LoRA should be applied
/// to your model. Different strategies might:
/// - Apply LoRA to all dense layers
/// - Apply LoRA only to layers with names matching a pattern
/// - Apply LoRA to all layers above a certain size
/// - Apply different LoRA ranks to different layer types
///
/// This gives you flexible control over how your model is adapted without hardcoding the logic.
/// </para>
/// </remarks>
public interface ILoRAConfiguration<T>
{
    /// <summary>
    /// Applies LoRA adaptation to a layer if applicable according to this configuration strategy.
    /// </summary>
    /// <param name="layer">The layer to potentially adapt with LoRA.</param>
    /// <returns>
    /// A LoRA-adapted version of the layer if the configuration determines it should be adapted,
    /// otherwise returns the original layer unchanged.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method examines the layer and decides whether to wrap it with a LoRA adapter.
    /// The decision can be based on:
    /// - Layer type (Dense, Convolutional, Attention, etc.)
    /// - Layer size or parameter count
    /// - Layer position in the model
    /// - Custom predicates or rules
    /// </para>
    /// <para><b>For Beginners:</b> This method looks at each layer in your model and decides:
    /// "Should I add LoRA to this layer?" If yes, it wraps the layer with a LoRA adapter.
    /// If no, it returns the layer as-is. This lets you selectively apply LoRA instead of
    /// adapting every single layer.
    /// </para>
    /// </remarks>
    ILayer<T> ApplyLoRA(ILayer<T> layer);

    /// <summary>
    /// Gets the rank of the low-rank decomposition to use for adapted layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The rank determines the number of parameters in the LoRA adaptation.
    /// Lower rank = fewer parameters = more efficient but less flexible.
    /// </para>
    /// <para>
    /// Common values:
    /// - 1-4: Minimal parameters, very efficient
    /// - 8: Good default balance
    /// - 16-32: More flexibility
    /// - 64+: Approaching full fine-tuning
    /// </para>
    /// </remarks>
    int Rank { get; }

    /// <summary>
    /// Gets the scaling factor (alpha) for LoRA adaptations.
    /// </summary>
    /// <remarks>
    /// Alpha controls how strongly LoRA adaptations affect outputs.
    /// Common practice: alpha = rank (for scaling factor of 1.0)
    /// Set to -1 to use rank as alpha (automatic scaling).
    /// </remarks>
    double Alpha { get; }

    /// <summary>
    /// Gets whether base layers should be frozen during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true (typical), only LoRA parameters are trained while base layer
    /// weights remain frozen. This dramatically reduces memory and compute requirements.
    /// </para>
    /// <para>
    /// When false, both base layer and LoRA parameters are trained. This uses more
    /// resources but may achieve better results in some scenarios.
    /// </para>
    /// </remarks>
    bool FreezeBaseLayer { get; }
}
