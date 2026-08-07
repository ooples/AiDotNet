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
/// This attribute exists because a tensor field is about to mean "parameter" by DEFAULT. Today the
/// framework cannot tell a weight from a cache — both are <c>Tensor&lt;T&gt;</c> — so it relies on
/// authors remembering to opt IN with <see cref="TrainableParameterAttribute"/>, and across the
/// library 2,197 tensor fields carry no classification at all. Opting in has failed empirically, so
/// the default inverts: a plain tensor field is a parameter, and the exceptions declare themselves.
/// </para>
/// <para>
/// Most scratch never needs this attribute. Nullable fields (<c>Tensor&lt;T&gt;?</c>) are already
/// excluded, which covers the usual cache shape — <c>_lastInput</c>, <c>_lastOutput</c> — and any
/// field whose name ends in <c>Gradient</c> is excluded by convention. Reach for
/// <c>[Scratch]</c> when a field is non-nullable but still not part of the model.
/// </para>
/// <para><b>For Beginners:</b> if deleting this tensor and rebuilding it on the next forward pass
/// would lose nothing, it is scratch.</para>
/// </remarks>
[AttributeUsage(AttributeTargets.Field, AllowMultiple = false, Inherited = false)]
public sealed class ScratchAttribute : Attribute
{
}
