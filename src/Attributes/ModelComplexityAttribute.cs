using AiDotNet.Enums;

namespace AiDotNet.Attributes;

/// <summary>
/// Specifies the computational complexity and resource requirements of a model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Apply this attribute to your model class to indicate how much
/// computing power it needs. This helps users choose models appropriate for their hardware.
/// </para>
/// <para>
/// <b>Usage:</b>
/// <code>
/// [ModelComplexity(ModelComplexity.High)]
/// public class ResNet&lt;T&gt; : NeuralNetworkBase&lt;T&gt; { }
/// </code>
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
public sealed class ModelComplexityAttribute : Attribute
{
    /// <summary>
    /// Gets the complexity level of this model.
    /// </summary>
    public ModelComplexity Complexity { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelComplexityAttribute"/> class.
    /// </summary>
    /// <param name="complexity">The computational complexity of this model.</param>
    public ModelComplexityAttribute(ModelComplexity complexity)
    {
        Complexity = complexity;
    }
}
