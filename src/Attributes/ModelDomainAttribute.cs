using AiDotNet.Enums;

namespace AiDotNet.Attributes;

/// <summary>
/// Specifies the application domain(s) that a model is designed for.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Apply this attribute to your model class to indicate what
/// field or industry it's best suited for. You can apply it multiple times if the
/// model works in several domains.
/// </para>
/// <para>
/// <b>Usage:</b>
/// <code>
/// [ModelDomain(ModelDomain.Vision)]
/// [ModelDomain(ModelDomain.Healthcare)]
/// public class MedicalImageClassifier&lt;T&gt; : NeuralNetworkBase&lt;T&gt; { }
/// </code>
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
public sealed class ModelDomainAttribute : Attribute
{
    /// <summary>
    /// Gets the application domain this model belongs to.
    /// </summary>
    public ModelDomain Domain { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelDomainAttribute"/> class.
    /// </summary>
    /// <param name="domain">The application domain for this model.</param>
    public ModelDomainAttribute(ModelDomain domain)
    {
        Domain = domain;
    }
}
