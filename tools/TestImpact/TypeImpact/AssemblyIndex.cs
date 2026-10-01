using System.Collections.Immutable;
using System.Reflection;
using System.Reflection.Metadata;
using System.Reflection.Metadata.Ecma335;
using System.Reflection.PortableExecutable;

namespace AiDotNet.TestImpact.TypeImpact;

/// <summary>One outermost type. Nested and compiler-generated types fold into it.</summary>
internal sealed class TypeNode
{
    public required string Key { get; init; }
    public required string Assembly { get; init; }
    /// <summary>Metadata full name of the outermost type, e.g. <c>AiDotNet.Models.ResNet`1</c>.</summary>
    public required string FullName { get; init; }
    public HashSet<TypeNode> References { get; } = [];
    public HashSet<TypeNode> ReferencedBy { get; } = [];
    public HashSet<string> Documents { get; } = new(StringComparer.Ordinal);
    /// <summary>Enumerates types by reflection (GetTypes and friends): its reach is not static.</summary>
    public bool Enumerates { get; set; }
    /// <summary>
    /// Declares a non-private const: consumers inline its value (ldc/ldstr, attribute blobs) and keep
    /// no token for this type, so a changed value is invisible to the reference graph.
    /// </summary>
    public bool DeclaresVisibleConstant { get; set; }
    public List<TestClass> TestClasses { get; } = [];
}

/// <summary>A concrete xUnit test class, possibly nested, with the tests it runs.</summary>
internal sealed class TestClass
{
    public required string VsTestName { get; init; }
    public required TypeNode Node { get; init; }
    public List<TestMethod> Methods { get; } = [];
}

/// <summary>A test method as VSTest names it, with its Category traits.</summary>
internal sealed record TestMethod(string FullyQualifiedName, IReadOnlySet<string> CertainCategories, IReadOnlySet<string> PossibleCategories);

/// <summary>
/// The type-reference graph over every assembly whose portable PDB places its sources under the
/// repository. Edges come from base types, interfaces, generic constraints, member signatures,
/// custom attributes (including typeof arguments), locals and every token in every method body.
/// </summary>
internal sealed class AssemblyIndex
{
    private readonly Dictionary<string, TypeNode> _nodes = new(StringComparer.Ordinal);
    private readonly Dictionary<string, List<TypeNode>> _byFullName = new(StringComparer.Ordinal);
    private readonly Dictionary<string, HashSet<TypeNode>> _byDocument = new(StringComparer.Ordinal);
    private readonly List<LoadedAssembly> _assemblies = [];

    public IReadOnlyCollection<TypeNode> Nodes => _nodes.Values;
    public IReadOnlyList<string> AssemblyNames => _assemblies.Select(a => a.Name).ToList();
    public IReadOnlySet<string> TestAssemblies => _assemblies.Where(a => a.ReferencesXunit).Select(a => a.Name).ToHashSet(StringComparer.Ordinal);

    public IReadOnlySet<TypeNode>? TypesInDocument(string repoRelativePath) =>
        _byDocument.TryGetValue(repoRelativePath, out var set) ? set : null;

    /// <summary>Every type with a source document whose key starts with <paramref name="prefix"/>.</summary>
    public HashSet<TypeNode> TypesInDocumentsUnder(string prefix)
    {
        var result = new HashSet<TypeNode>();
        foreach (var (document, types) in _byDocument)
        {
            if (document.StartsWith(prefix, StringComparison.Ordinal)) result.UnionWith(types);
        }
        return result;
    }

    public static AssemblyIndex Load(IEnumerable<string> binDirectories, string repoRoot)
    {
        var index = new AssemblyIndex();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var dir in binDirectories)
        {
            foreach (var dll in Directory.EnumerateFiles(dir, "*.dll"))
            {
                var pdb = Path.ChangeExtension(dll, ".pdb");
                if (!File.Exists(pdb) || !seen.Add(Path.GetFileName(dll)))
                {
                    continue;
                }

                var loaded = LoadedAssembly.TryOpen(dll, pdb, repoRoot);
                if (loaded is not null)
                {
                    index._assemblies.Add(loaded);
                }
            }
        }

        if (index._assemblies.Count == 0)
        {
            throw new InvalidOperationException("no assembly in the given bin directories has a portable PDB under the repository");
        }

        foreach (var asm in index._assemblies)
        {
            index.CreateNodes(asm);
        }

        foreach (var asm in index._assemblies)
        {
            index.CollectEdges(asm);
        }

        foreach (var asm in index._assemblies.Where(a => a.ReferencesXunit))
        {
            index.CollectTests(asm);
        }

        return index;
    }

    private void CreateNodes(LoadedAssembly asm)
    {
        var md = asm.Metadata;
        foreach (var handle in md.TypeDefinitions)
        {
            var outer = Outermost(md, handle);
            var fullName = FullName(md, outer);
            var key = asm.Name + "!" + fullName;
            if (!_nodes.TryGetValue(key, out var node))
            {
                node = new TypeNode { Key = key, Assembly = asm.Name, FullName = fullName };
                _nodes.Add(key, node);
                if (!_byFullName.TryGetValue(fullName, out var list))
                {
                    _byFullName[fullName] = list = [];
                }

                list.Add(node);
            }

            asm.NodeOf[handle] = node;
            if (DeclaresVisibleConstant(md, handle))
            {
                node.DeclaresVisibleConstant = true;
            }
        }

        foreach (var (typeHandle, documents) in asm.DocumentsByType())
        {
            var node = asm.NodeOf[typeHandle];
            foreach (var doc in documents)
            {
                node.Documents.Add(doc);
                if (!_byDocument.TryGetValue(doc, out var set))
                {
                    _byDocument[doc] = set = [];
                }

                set.Add(node);
            }
        }
    }

    private static bool DeclaresVisibleConstant(MetadataReader md, TypeDefinitionHandle handle)
    {
        var type = md.GetTypeDefinition(handle);
        // Enum members are literals too, but every consumer names the enum type in a signature.
        if (type.BaseType.Kind == HandleKind.TypeReference &&
            md.GetString(md.GetTypeReference((TypeReferenceHandle)type.BaseType).Name) == "Enum")
        {
            return false;
        }

        foreach (var attributes in type.GetFields().Select(f => md.GetFieldDefinition(f).Attributes))
        {
            if ((attributes & FieldAttributes.Literal) != 0 &&
                (attributes & FieldAttributes.FieldAccessMask) != FieldAttributes.Private)
            {
                return true;
            }
        }

        return false;
    }

    private void CollectEdges(LoadedAssembly asm)
    {
        var md = asm.Metadata;
        var provider = new ReferenceCollector(this, asm);
        foreach (var handle in md.TypeDefinitions)
        {
            var node = asm.NodeOf[handle];
            provider.Target = node;
            var type = md.GetTypeDefinition(handle);
            provider.AddHandle(type.BaseType);
            foreach (var iface in type.GetInterfaceImplementations())
            {
                provider.AddHandle(md.GetInterfaceImplementation(iface).Interface);
            }

            AddGenericConstraints(md, type.GetGenericParameters(), provider);
            AddAttributes(md, type.GetCustomAttributes(), provider);

            foreach (var field in type.GetFields().Select(md.GetFieldDefinition))
            {
                field.DecodeSignature(provider, null);
                AddAttributes(md, field.GetCustomAttributes(), provider);
            }

            foreach (var propertyHandle in type.GetProperties())
            {
                AddAttributes(md, md.GetPropertyDefinition(propertyHandle).GetCustomAttributes(), provider);
            }

            foreach (var method in type.GetMethods().Select(md.GetMethodDefinition))
            {
                method.DecodeSignature(provider, null);
                AddGenericConstraints(md, method.GetGenericParameters(), provider);
                AddAttributes(md, method.GetCustomAttributes(), provider);
                foreach (var parameterHandle in method.GetParameters())
                {
                    AddAttributes(md, md.GetParameter(parameterHandle).GetCustomAttributes(), provider);
                }

                if (method.RelativeVirtualAddress == 0)
                {
                    continue;
                }

                var body = asm.PE.GetMethodBody(method.RelativeVirtualAddress);
                if (!body.LocalSignature.IsNil)
                {
                    md.GetStandaloneSignature(body.LocalSignature).DecodeLocalSignature(provider, null);
                }

                foreach (var token in IlTokens.Scan(body))
                {
                    provider.AddHandle(token);
                }
            }
        }
    }

    private static void AddGenericConstraints(MetadataReader md, GenericParameterHandleCollection parameters, ReferenceCollector provider)
    {
        foreach (var parameterHandle in parameters)
        {
            foreach (var constraintHandle in md.GetGenericParameter(parameterHandle).GetConstraints())
            {
                provider.AddHandle(md.GetGenericParameterConstraint(constraintHandle).Type);
            }
        }
    }

    private static void AddAttributes(MetadataReader md, CustomAttributeHandleCollection attributes, ReferenceCollector provider)
    {
        foreach (var attribute in attributes.Select(md.GetCustomAttribute))
        {
            provider.AddHandle(attribute.Constructor);
            try
            {
                // typeof(X) arguments arrive as serialized names; the provider resolves them.
                attribute.DecodeValue(provider);
            }
            catch (Exception e) when (e is BadImageFormatException or ArgumentException or InvalidOperationException)
            {
                // An argument whose enum type lives outside the loaded set cannot be sized; the
                // constructor edge above already names the attribute itself.
            }
        }
    }

    /// <summary>Resolves a type name as the runtime would see it to the loaded node(s).</summary>
    public IEnumerable<TypeNode> Resolve(string outermostFullName) =>
        _byFullName.TryGetValue(outermostFullName, out var list) ? list : [];

    private void CollectTests(LoadedAssembly asm)
    {
        var md = asm.Metadata;
        foreach (var handle in md.TypeDefinitions)
        {
            var type = md.GetTypeDefinition(handle);
            if ((type.Attributes & TypeAttributes.Abstract) != 0 || type.GetGenericParameters().Count > 0 ||
                (type.Attributes & TypeAttributes.VisibilityMask) is not (TypeAttributes.Public or TypeAttributes.NestedPublic))
            {
                continue;
            }

            var vsTestName = VsTestTypeName(md, handle);
            var classCategories = Categories(asm, type.GetCustomAttributes());
            var methods = new List<TestMethod>();
            var seenNames = new HashSet<string>(StringComparer.Ordinal);
            var possibleFromBases = new HashSet<string>(StringComparer.Ordinal);
            // Own methods first, then each base in turn: an override hides the base declaration.
            foreach (var (declaringAsm, declaringHandle, isOwn) in SelfAndBases(asm, handle))
            {
                var declaring = declaringAsm.Metadata.GetTypeDefinition(declaringHandle);
                if (!isOwn)
                {
                    possibleFromBases.UnionWith(Categories(declaringAsm, declaring.GetCustomAttributes()));
                }

                foreach (var methodHandle in declaring.GetMethods())
                {
                    var method = declaringAsm.Metadata.GetMethodDefinition(methodHandle);
                    var name = declaringAsm.Metadata.GetString(method.Name);
                    if (!IsTestMethod(declaringAsm, method) || !seenNames.Add(name))
                    {
                        continue;
                    }

                    var certain = new HashSet<string>(classCategories, StringComparer.Ordinal);
                    certain.UnionWith(Categories(declaringAsm, method.GetCustomAttributes()));
                    var possible = new HashSet<string>(certain, StringComparer.Ordinal);
                    possible.UnionWith(possibleFromBases);
                    methods.Add(new TestMethod(vsTestName + "." + name, certain, possible));
                }
            }

            if (methods.Count == 0)
            {
                continue;
            }

            var testClass = new TestClass { VsTestName = vsTestName, Node = asm.NodeOf[handle] };
            testClass.Methods.AddRange(methods);
            testClass.Node.TestClasses.Add(testClass);
        }
    }

    private IEnumerable<(LoadedAssembly, TypeDefinitionHandle, bool)> SelfAndBases(LoadedAssembly asm, TypeDefinitionHandle handle)
    {
        var currentAsm = asm;
        var current = handle;
        bool own = true;
        for (int depth = 0; depth < 64; depth++)
        {
            yield return (currentAsm, current, own);
            own = false;
            var next = ResolveDefinition(currentAsm, currentAsm.Metadata.GetTypeDefinition(current).BaseType);
            if (next is null)
            {
                yield break;
            }

            (currentAsm, current) = next.Value;
        }
    }

    /// <summary>Resolves a base-type handle to its definition when it is one of ours.</summary>
    private (LoadedAssembly, TypeDefinitionHandle)? ResolveDefinition(LoadedAssembly asm, EntityHandle handle)
    {
        var md = asm.Metadata;
        switch (handle.Kind)
        {
            case HandleKind.TypeDefinition:
                return (asm, (TypeDefinitionHandle)handle);
            case HandleKind.TypeSpecification:
                var blob = md.GetBlobReader(md.GetTypeSpecification((TypeSpecificationHandle)handle).Signature);
                if (blob.ReadSignatureTypeCode() != SignatureTypeCode.GenericTypeInstance)
                {
                    return null;
                }

                blob.ReadCompressedInteger(); // class or valuetype
                return ResolveDefinition(asm, blob.ReadTypeHandle());
            case HandleKind.TypeReference:
                var name = ReferenceFullName(md, (TypeReferenceHandle)handle);
                foreach (var other in _assemblies)
                {
                    if (other.DefinitionsByFullName.TryGetValue(name, out var definition))
                    {
                        return (other, definition);
                    }
                }

                return null;
            default:
                return null;
        }
    }

    private bool IsTestMethod(LoadedAssembly asm, MethodDefinition method)
    {
        foreach (var attributeHandle in method.GetCustomAttributes())
        {
            var attributeType = AttributeTypeName(asm, asm.Metadata.GetCustomAttribute(attributeHandle));
            if (attributeType is null)
            {
                continue;
            }

            // Walk our own attribute hierarchy: a repository [HeavyFact] derives from FactAttribute.
            if (IsFactName(attributeType.Value.Name) || InheritsFact(attributeType.Value.Asm, attributeType.Value.Definition))
            {
                return true;
            }
        }

        return false;
    }

    private static bool IsFactName(string name) =>
        name.EndsWith("FactAttribute", StringComparison.Ordinal) || name.EndsWith("TheoryAttribute", StringComparison.Ordinal);

    private bool InheritsFact(LoadedAssembly? asm, TypeDefinitionHandle? definition)
    {
        if (asm is null || definition is null)
        {
            return false;
        }

        foreach (var (baseAsm, baseHandle, _) in SelfAndBases(asm, definition.Value))
        {
            var baseType = baseAsm.Metadata.GetTypeDefinition(baseHandle);
            if (IsFactName(baseAsm.Metadata.GetString(baseType.Name)))
            {
                return true;
            }

            if (baseType.BaseType.Kind == HandleKind.TypeReference &&
                IsFactName(baseAsm.Metadata.GetString(baseAsm.Metadata.GetTypeReference((TypeReferenceHandle)baseType.BaseType).Name)))
            {
                return true;
            }
        }

        return false;
    }

    private (string Name, LoadedAssembly? Asm, TypeDefinitionHandle? Definition)? AttributeTypeName(LoadedAssembly asm, CustomAttribute attribute)
    {
        var md = asm.Metadata;
        EntityHandle parent;
        if (attribute.Constructor.Kind == HandleKind.MemberReference)
        {
            parent = md.GetMemberReference((MemberReferenceHandle)attribute.Constructor).Parent;
        }
        else if (attribute.Constructor.Kind == HandleKind.MethodDefinition)
        {
            parent = md.GetMethodDefinition((MethodDefinitionHandle)attribute.Constructor).GetDeclaringType();
        }
        else
        {
            return null;
        }

        if (parent.Kind == HandleKind.TypeDefinition)
        {
            var definition = (TypeDefinitionHandle)parent;
            return (md.GetString(md.GetTypeDefinition(definition).Name), asm, definition);
        }

        if (parent.Kind == HandleKind.TypeReference)
        {
            var reference = md.GetTypeReference((TypeReferenceHandle)parent);
            var resolved = ResolveDefinition(asm, parent);
            return (md.GetString(reference.Name), resolved?.Item1, resolved?.Item2);
        }

        return null;
    }

    /// <summary>Category values of every [Trait("Category", value)] in the collection.</summary>
    private HashSet<string> Categories(LoadedAssembly asm, CustomAttributeHandleCollection attributes)
    {
        var result = new HashSet<string>(StringComparer.Ordinal);
        foreach (var attribute in attributes.Select(asm.Metadata.GetCustomAttribute))
        {
            var name = AttributeTypeName(asm, attribute);
            if (name?.Name != "TraitAttribute")
            {
                continue;
            }

            var value = attribute.DecodeValue(new StringOnlyProvider());
            if (value.FixedArguments.Length == 2 &&
                value.FixedArguments[0].Value is "Category" &&
                value.FixedArguments[1].Value is string category)
            {
                result.Add(category);
            }
        }

        return result;
    }

    internal void AddEdge(TypeNode from, TypeNode to)
    {
        if (!ReferenceEquals(from, to) && from.References.Add(to))
        {
            to.ReferencedBy.Add(from);
        }
    }

    internal static TypeDefinitionHandle Outermost(MetadataReader md, TypeDefinitionHandle handle)
    {
        var current = handle;
        while (true)
        {
            var declaring = md.GetTypeDefinition(current).GetDeclaringType();
            if (declaring.IsNil)
            {
                return current;
            }

            current = declaring;
        }
    }

    internal static string FullName(MetadataReader md, TypeDefinitionHandle handle)
    {
        var type = md.GetTypeDefinition(handle);
        var ns = md.GetString(type.Namespace);
        var name = md.GetString(type.Name);
        return ns.Length == 0 ? name : ns + "." + name;
    }

    /// <summary>VSTest's class name for a test type: namespace, then '+' between nesting levels.</summary>
    private static string VsTestTypeName(MetadataReader md, TypeDefinitionHandle handle)
    {
        var type = md.GetTypeDefinition(handle);
        var declaring = type.GetDeclaringType();
        var name = md.GetString(type.Name);
        return declaring.IsNil ? FullName(md, handle) : VsTestTypeName(md, declaring) + "+" + name;
    }

    internal static string ReferenceFullName(MetadataReader md, TypeReferenceHandle handle)
    {
        var reference = md.GetTypeReference(handle);
        // A nested reference is scoped by its declaring reference; collapse to the outermost.
        while (reference.ResolutionScope.Kind == HandleKind.TypeReference)
        {
            reference = md.GetTypeReference((TypeReferenceHandle)reference.ResolutionScope);
        }

        var ns = md.GetString(reference.Namespace);
        var name = md.GetString(reference.Name);
        return ns.Length == 0 ? name : ns + "." + name;
    }

    /// <summary>Decodes only primitive and string attribute arguments (Trait values).</summary>
    private sealed class StringOnlyProvider : ICustomAttributeTypeProvider<object?>
    {
        public object? GetPrimitiveType(PrimitiveTypeCode typeCode) => null;
        public object? GetSystemType() => null;
        public object? GetSZArrayType(object? elementType) => null;
        public object? GetTypeFromDefinition(MetadataReader reader, TypeDefinitionHandle handle, byte rawTypeKind) => null;
        public object? GetTypeFromReference(MetadataReader reader, TypeReferenceHandle handle, byte rawTypeKind) => null;
        public object? GetTypeFromSerializedName(string name) => null;
        public PrimitiveTypeCode GetUnderlyingEnumType(object? type) => PrimitiveTypeCode.Int32;
        public bool IsSystemType(object? type) => false;
    }
}
