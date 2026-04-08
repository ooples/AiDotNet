using AiDotNet.Enums;

namespace AiDotNet.Attributes;

/// <summary>
/// Specifies the type of an AI pipeline component (Tier 2 metadata).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Apply this attribute to components that are part of an AI pipeline
/// but aren't standalone ML models. Components transform, route, or process data.
/// You can apply it multiple times if a component serves multiple roles.
/// </para>
/// <para>
/// <b>Usage:</b>
/// <code>
/// [ComponentType(ComponentType.Retriever)]
/// [PipelineStage(PipelineStage.Retrieval)]
/// public class HybridRetriever&lt;T&gt; : RetrieverBase&lt;T&gt; { }
/// </code>
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
public sealed class ComponentTypeAttribute : Attribute
{
    /// <summary>
    /// Gets the component type.
    /// </summary>
    public ComponentType Type { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ComponentTypeAttribute"/> class.
    /// </summary>
    /// <param name="type">The type of component.</param>
    public ComponentTypeAttribute(ComponentType type)
    {
        Type = type;
    }
}
