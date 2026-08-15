namespace AiDotNet.Attributes;

/// <summary>
/// Declares the input shape a composite layer feeds to one of its sub-layers, so the shape can be
/// generated rather than hand-written.
/// </summary>
/// <remarks>
/// <para>
/// A composite's children do not all receive the composite's own input. A transformer encoder feeds
/// its embedding width to the norms and to the first feed-forward, but the FEED-FORWARD width to the
/// second one, and self-attention wants a sequence axis in front. Only the composite knows that, and
/// it is the single fact <c>LayerBase.BringUpDeclaredSubLayers</c> needs in order to bring a child up
/// to shapes or all the way to weights on the composite's behalf.
/// </para>
/// <para>
/// Written as a comma-separated list of C# expressions evaluated in the composite's own scope, the
/// same form <c>[TrainableParameter(Shape = "...")]</c> uses:
/// <code>
/// [SubLayerInput("1, _embeddingSize")] private MultiHeadAttentionLayer&lt;T&gt; _selfAttention;
/// [SubLayerInput("_embeddingSize")]    private LayerNormalizationLayer&lt;T&gt; _norm1;
/// [SubLayerInput("_feedForwardDim")]   private FeedForwardLayer&lt;T&gt; _feedForward2;
/// </code>
/// The generator emits <c>DeclaredSubLayerShapes()</c> from these, so no composite implements it.
/// </para>
/// <para>
/// The generated method answers empty while any child is still null or any declared axis is still
/// negative — a composite builds its children inside its initializer, so both are ordinary states
/// before that runs, and an empty declaration is the base's signal that the layer cannot say yet.
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Field, AllowMultiple = false, Inherited = false)]
public sealed class SubLayerInputAttribute : Attribute
{
    /// <summary>
    /// Creates the declaration.
    /// </summary>
    /// <param name="shape">
    /// Comma-separated axis expressions for the shape this sub-layer receives, evaluated in the
    /// declaring composite's scope — for example <c>"1, _embeddingSize"</c>.
    /// </param>
    public SubLayerInputAttribute(string shape)
    {
        Shape = shape;
    }

    /// <summary>The declared input shape for this sub-layer.</summary>
    public string Shape { get; }
}
