namespace AiDotNet.Attributes;

/// <summary>
/// Marks a tensor field as transient working state: not trained, not counted, not saved.
/// </summary>
/// <remarks>
/// <para>
/// Scratch is everything a layer keeps for its own convenience — a cached activation held for the
/// backward pass, an intermediate buffer reused across calls, a workspace for a fused kernel. None
/// of it is part of the model; recreating it from scratch after a load changes nothing.
/// </para>
/// <para>
/// The framework deliberately does not infer a role from <c>Tensor&lt;T&gt;</c>. A cache and a weight have
/// the same storage type, so inference would silently hand activations to an optimizer. This
/// declaration tells the shared analyzer and generators that the storage is transient.
/// </para>
/// <para>
/// Nullability is only a storage fact and does not imply scratch. Apply <c>[Scratch]</c> to cached
/// activations and workspaces whether nullable or non-nullable. A matching generated gradient field
/// is the one narrow convention recognized because its owning trainable parameter supplies the
/// semantic relationship.
/// </para>
/// <para><b>For Beginners:</b> if deleting this tensor and rebuilding it on the next forward pass
/// would lose nothing, it is scratch.</para>
/// </remarks>
[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
public sealed class ScratchAttribute : Attribute
{
}
