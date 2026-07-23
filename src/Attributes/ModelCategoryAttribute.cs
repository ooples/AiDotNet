using AiDotNet.Enums;

namespace AiDotNet.Attributes;

/// <summary>
/// Specifies the algorithm family or category that a model belongs to.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Apply this attribute to your model class to indicate what kind
/// of algorithm it uses. You can apply it multiple times if the model combines
/// multiple algorithm families (e.g., a Transformer that is also an Autoencoder).
/// </para>
/// <para>
/// <b>Usage:</b>
/// <code>
/// [ModelCategory(ModelCategory.Transformer)]
/// [ModelCategory(ModelCategory.Autoencoder)]
/// public class AudioMAE&lt;T&gt; : NeuralNetworkBase&lt;T&gt; { }
/// </code>
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
public sealed class ModelCategoryAttribute : Attribute
{
    /// <summary>
    /// Gets the algorithm category this model belongs to.
    /// </summary>
    public ModelCategory Category { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelCategoryAttribute"/> class.
    /// </summary>
    /// <param name="category">The algorithm category for this model.</param>
    public ModelCategoryAttribute(ModelCategory category)
    {
        Category = category;
    }
}
