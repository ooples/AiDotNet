namespace AiDotNet.Attributes;

/// <summary>
/// Requests a double-precision generated layer-test scaffold.
/// </summary>
/// <remarks>
/// This is generator plumbing for numerically conditioned finite-difference invariants. It does
/// not change the layer's production precision or relax any invariant; it prevents float32
/// representational noise from being mistaken for a derivative of the underlying real-valued
/// function.
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
internal sealed class GenerateDoubleTestScaffoldAttribute : Attribute
{
}
