namespace AiDotNet.Attributes;

/// <summary>
/// Specifies the expected input and output types for a model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Apply this attribute to your model class to declare what type of
/// data it expects as input and what it produces as output. This makes it easy to discover
/// which models work with your data format without reading documentation.
/// </para>
/// <para>
/// <b>Usage:</b>
/// <code>
/// [ModelInput(typeof(Tensor&lt;&gt;), typeof(Vector&lt;&gt;))]
/// public class ResNet&lt;T&gt; : NeuralNetworkBase&lt;T&gt; { }
/// </code>
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
public sealed class ModelInputAttribute : Attribute
{
    /// <summary>
    /// Gets the expected input type for this model.
    /// </summary>
    public Type InputType { get; }

    /// <summary>
    /// Gets the expected output type for this model.
    /// </summary>
    public Type OutputType { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelInputAttribute"/> class.
    /// </summary>
    /// <param name="inputType">The expected input type (e.g., typeof(Tensor&lt;&gt;)).</param>
    /// <param name="outputType">The expected output type (e.g., typeof(Vector&lt;&gt;)).</param>
    public ModelInputAttribute(Type inputType, Type outputType)
    {
        InputType = inputType;
        OutputType = outputType;
    }
}
