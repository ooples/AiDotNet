namespace AiDotNet.Attributes;

/// <summary>
/// Marks a class that implements IFullModel as exempt from model metadata validation diagnostics.
/// </summary>
/// <remarks>
/// <para>
/// Apply this attribute to classes that implement IFullModel but are not user-facing models
/// intended for discovery. For example, result wrappers, adapted model containers, or
/// internal infrastructure classes that should not appear in the model discovery API.
/// </para>
/// <para>
/// <b>For Beginners:</b> Some classes implement IFullModel because they need to wrap or
/// contain models, but they aren't models themselves. This attribute tells the build system
/// to skip validation checks (like requiring [ModelDomain], [ModelCategory], etc.) on these
/// classes.
/// </para>
/// <para>
/// <b>Usage:</b>
/// <code>
/// [ModelMetadataExempt]
/// public class AiModelResult&lt;T, TInput, TOutput&gt; : IFullModel&lt;T, TInput, TOutput&gt; { }
/// </code>
/// </para>
/// </remarks>
/// <remarks>
/// This attribute is intentionally public so that external consumers extending the library
/// can exempt their own IFullModel implementations from metadata validation diagnostics.
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
public sealed class ModelMetadataExemptAttribute : Attribute
{
}
