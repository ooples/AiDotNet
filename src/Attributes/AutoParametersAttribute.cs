namespace AiDotNet.Attributes;

/// <summary>
/// Marks a layer that has migrated to generated parameter plumbing.
/// </summary>
/// <remarks>
/// <para>
/// This attribute does not classify fields. A raw <c>Tensor&lt;T&gt;</c> can be a weight, fitted state,
/// buffer, scratch cache, alias, or externally owned storage; its CLR type, name, and nullability
/// cannot choose safely. Each numeric-state declaration therefore carries one explicit semantic
/// attribute, and the generator supplies the common count/read/write/restore plumbing.
/// </para>
/// <para>
/// It remains as a source-compatible migration marker while existing layers move to the exhaustive
/// semantic declarations. The generator itself is driven by those member declarations, not by this
/// class marker.
/// </para>
/// <para><b>For Beginners:</b> identify what each numeric field means; the framework then generates
/// all repetitive parameter plumbing.</para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
public sealed class AutoParametersAttribute : Attribute
{
}
