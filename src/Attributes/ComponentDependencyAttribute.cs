namespace AiDotNet.Attributes;

/// <summary>
/// Declares a dependency that a component requires from another component or interface.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This tells the pipeline builder what other components this one needs.
/// For example, a retriever might depend on an embedding model, or a reranker might depend on
/// a retriever. The pipeline builder uses this information to validate composition.
/// </para>
/// <para>
/// <b>Usage:</b>
/// <code>
/// [ComponentType(ComponentType.Retriever)]
/// [ComponentDependency(typeof(IEmbeddingModel&lt;&gt;), "Embedding model for dense retrieval")]
/// public class DenseRetriever&lt;T&gt; : RetrieverBase&lt;T&gt; { }
/// </code>
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
public sealed class ComponentDependencyAttribute : Attribute
{
    /// <summary>
    /// Gets the type (typically an interface) that this component depends on.
    /// </summary>
    public Type DependencyType { get; }

    /// <summary>
    /// Gets or sets a description of why this dependency is needed.
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets whether this dependency is required (true) or optional (false).
    /// Default is true (required).
    /// </summary>
    public bool Required { get; set; } = true;

    /// <summary>
    /// Initializes a new instance of the <see cref="ComponentDependencyAttribute"/> class.
    /// </summary>
    /// <param name="dependencyType">The type that this component depends on.</param>
    public ComponentDependencyAttribute(Type dependencyType)
    {
        DependencyType = dependencyType;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ComponentDependencyAttribute"/> class.
    /// </summary>
    /// <param name="dependencyType">The type that this component depends on.</param>
    /// <param name="description">Description of why this dependency is needed.</param>
    public ComponentDependencyAttribute(Type dependencyType, string description)
    {
        DependencyType = dependencyType ?? throw new ArgumentNullException(nameof(dependencyType));
        if (string.IsNullOrWhiteSpace(description))
            throw new ArgumentException("Dependency description must not be empty.", nameof(description));
        Description = description;
    }
}
