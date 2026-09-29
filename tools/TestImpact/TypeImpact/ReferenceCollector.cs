using System.Collections.Immutable;
using System.Reflection.Metadata;

namespace AiDotNet.TestImpact.TypeImpact;

/// <summary>
/// Records an edge from <see cref="Target"/> to every loaded type a handle, signature or attribute
/// argument names. It is a signature provider for its side effect; the decoded values are unused.
/// </summary>
internal sealed class ReferenceCollector :
    ISignatureTypeProvider<object?, object?>,
    ICustomAttributeTypeProvider<object?>
{
    // Reflection entry points whose result is a set of types no signature names.
    private static readonly HashSet<string> EnumeratingMembers = new(StringComparer.Ordinal)
    {
        "System.Reflection.Assembly::GetTypes",
        "System.Reflection.Assembly::GetExportedTypes",
        "System.Reflection.Assembly::get_DefinedTypes",
        "System.Reflection.Assembly::get_ExportedTypes",
        "System.Reflection.Module::GetTypes",
    };

    private readonly AssemblyIndex _index;
    private readonly LoadedAssembly _asm;

    public ReferenceCollector(AssemblyIndex index, LoadedAssembly asm)
    {
        _index = index;
        _asm = asm;
    }

    public TypeNode Target { get; set; } = null!;

    private MetadataReader Md => _asm.Metadata;

    public void AddHandle(EntityHandle handle)
    {
        if (handle.IsNil)
        {
            return;
        }

        switch (handle.Kind)
        {
            case HandleKind.TypeDefinition:
                Link(_asm.NodeOf[(TypeDefinitionHandle)handle]);
                break;
            case HandleKind.TypeReference:
                LinkReference((TypeReferenceHandle)handle);
                break;
            case HandleKind.TypeSpecification:
                Md.GetTypeSpecification((TypeSpecificationHandle)handle).DecodeSignature(this, null);
                break;
            case HandleKind.MethodDefinition:
                Link(_asm.NodeOf[Md.GetMethodDefinition((MethodDefinitionHandle)handle).GetDeclaringType()]);
                break;
            case HandleKind.FieldDefinition:
                Link(_asm.NodeOf[Md.GetFieldDefinition((FieldDefinitionHandle)handle).GetDeclaringType()]);
                break;
            case HandleKind.MethodSpecification:
                var spec = Md.GetMethodSpecification((MethodSpecificationHandle)handle);
                AddHandle(spec.Method);
                spec.DecodeSignature(this, null);
                break;
            case HandleKind.MemberReference:
                var member = Md.GetMemberReference((MemberReferenceHandle)handle);
                AddHandle(member.Parent);
                NoteEnumeration(member);
                if (member.GetKind() == MemberReferenceKind.Method)
                {
                    member.DecodeMethodSignature(this, null);
                }
                else
                {
                    member.DecodeFieldSignature(this, null);
                }

                break;
        }
    }

    private void NoteEnumeration(MemberReference member)
    {
        if (member.Parent.Kind != HandleKind.TypeReference)
        {
            return;
        }

        var owner = AssemblyIndex.ReferenceFullName(Md, (TypeReferenceHandle)member.Parent);
        if (EnumeratingMembers.Contains(owner + "::" + Md.GetString(member.Name)))
        {
            Target.Enumerates = true;
        }
    }

    private void Link(TypeNode node) => _index.AddEdge(Target, node);

    private void LinkReference(TypeReferenceHandle handle)
    {
        foreach (var node in _index.Resolve(AssemblyIndex.ReferenceFullName(Md, handle)))
        {
            Link(node);
        }
    }

    // ISignatureTypeProvider: every type-bearing callback links; composites return nothing.
    public object? GetTypeFromDefinition(MetadataReader reader, TypeDefinitionHandle handle, byte rawTypeKind)
    {
        Link(_asm.NodeOf[handle]);
        return null;
    }

    public object? GetTypeFromReference(MetadataReader reader, TypeReferenceHandle handle, byte rawTypeKind)
    {
        LinkReference(handle);
        return null;
    }

    public object? GetTypeFromSpecification(MetadataReader reader, object? genericContext, TypeSpecificationHandle handle, byte rawTypeKind)
    {
        reader.GetTypeSpecification(handle).DecodeSignature(this, genericContext);
        return null;
    }

    public object? GetSerializedName(string name) => GetTypeFromSerializedName(name);

    public object? GetTypeFromSerializedName(string name)
    {
        // "Ns.Outer+Inner`1[[Arg, Asm]], Asm" -> "Ns.Outer" plus each generic argument.
        foreach (var part in SplitSerializedName(name))
        {
            foreach (var node in _index.Resolve(part))
            {
                Link(node);
            }
        }

        return null;
    }

    internal static IEnumerable<string> SplitSerializedName(string name)
    {
        var parts = new List<string>();
        int depth = 0;
        var current = new System.Text.StringBuilder();
        bool inAssembly = false;
        foreach (char c in name)
        {
            switch (c)
            {
                case '[':
                    if (current.Length > 0 && !inAssembly) { parts.Add(current.ToString()); }
                    current.Clear();
                    inAssembly = false;
                    depth++;
                    break;
                case ']':
                    if (current.Length > 0 && !inAssembly) { parts.Add(current.ToString()); }
                    current.Clear();
                    inAssembly = false;
                    depth--;
                    break;
                case ',':
                    if (current.Length > 0 && !inAssembly) { parts.Add(current.ToString()); }
                    current.Clear();
                    inAssembly = true;
                    break;
                default:
                    if (!inAssembly) { current.Append(c); }
                    break;
            }
        }

        if (current.Length > 0 && !inAssembly)
        {
            parts.Add(current.ToString());
        }

        return parts
            .Select(p => p.Trim())
            .Where(p => p.Length > 0)
            .Select(p => p.Split('+')[0]);
    }

    public object? GetPrimitiveType(PrimitiveTypeCode typeCode) => null;
    public object? GetSZArrayType(object? elementType) => null;
    public object? GetArrayType(object? elementType, ArrayShape shape) => null;
    public object? GetByReferenceType(object? elementType) => null;
    public object? GetPointerType(object? elementType) => null;
    public object? GetPinnedType(object? elementType) => null;
    public object? GetGenericInstantiation(object? genericType, ImmutableArray<object?> typeArguments) => null;
    public object? GetGenericMethodParameter(object? genericContext, int index) => null;
    public object? GetGenericTypeParameter(object? genericContext, int index) => null;
    public object? GetFunctionPointerType(MethodSignature<object?> signature) => null;
    public object? GetModifiedType(object? modifier, object? unmodifiedType, bool isRequired) => null;
    public object? GetSystemType() => null;
    public bool IsSystemType(object? type) => false;

    // Unknown enum argument types are sized as Int32, the C# default underlying type.
    public PrimitiveTypeCode GetUnderlyingEnumType(object? type) => PrimitiveTypeCode.Int32;
}
