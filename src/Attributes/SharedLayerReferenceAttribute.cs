namespace AiDotNet.Attributes;

/// <summary>
/// Marks a layer field that refers to a layer another part of the network owns, such as an output head that shares the
/// token embedding's table. The referenced layer is registered where it is owned, so it is not a child of the layer
/// holding the reference: registering it there as well would count its weights twice.
/// </summary>
[AttributeUsage(AttributeTargets.Field, AllowMultiple = false, Inherited = false)]
public sealed class SharedLayerReferenceAttribute : Attribute
{
}
