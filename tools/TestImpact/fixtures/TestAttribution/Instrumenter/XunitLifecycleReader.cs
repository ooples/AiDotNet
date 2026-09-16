using System.Security.Cryptography;
using System.Text;
using AiDotNet.TestImpact;
using Mono.Cecil;

internal sealed record XunitLifecycleResult(SourceLifecycleMap Map, SourceMethod[] SyntheticMethods);

// Model runner entry points, not every helper in the test assembly. Ordinary
// helpers remain reachable through IL calls; unknown runner extensions remain
// explicit open roots. Data callbacks remain group roots unless the bound
// framework enforces deferred discovery; custom discoverers remain open.
internal static class XunitLifecycleReader
{
    internal static XunitLifecycleResult Read(AssemblyDefinition assembly)
    {
        TypeDefinition[] types = AllTypes(assembly.MainModule.Types).ToArray();
        var groups = new HashSet<string>(StringComparer.Ordinal);
        var tests = new List<SourceTestLifecycle>();
        var synthetic = new Dictionary<string, SourceMethod>(StringComparer.Ordinal);
        bool deferredTheories = assembly.CustomAttributes
            .Where(attribute => attribute.AttributeType.FullName == "Xunit.TestFrameworkAttribute")
            .Select(attribute => NamedType(attribute, assembly))
            .Any(framework => Hierarchy(framework).Any(type =>
                type.FullName == "AiDotNet.TestImpact.Xunit.AttributionTestFramework" &&
                type.Module.Assembly.Name.Name == "Attribution.Xunit" &&
                type.Methods.Any(method => method.Name == "CreateDiscoverer" && method.IsFinal)));
        foreach (TypeDefinition type in types.Where(type => type.Name == "<Module>"))
            foreach (MethodDefinition method in type.Methods.Where(method => method.IsConstructor)) groups.Add(Id(method));

        foreach (CustomAttribute attribute in assembly.CustomAttributes)
        {
            AddBeforeAfter(attribute, groups);
            if (attribute.AttributeType.FullName == "Xunit.TestFrameworkAttribute")
            {
                TypeDefinition? framework = NamedType(attribute, assembly);
                if (framework is null) groups.Add("unresolved:xunit:framework");
                else AddFramework(framework, groups);
            }
            else if (attribute.AttributeType.FullName is "Xunit.TestCaseOrdererAttribute" or "Xunit.TestCollectionOrdererAttribute")
                AddAll(NamedType(attribute, assembly), groups, "assembly-orderer");
            else if (attribute.AttributeType.FullName == "Xunit.CollectionBehaviorAttribute" &&
                     attribute.ConstructorArguments.Any(argument => argument.Value is TypeReference))
                groups.Add("unresolved:xunit:collection-factory");
        }

        foreach (TypeDefinition type in types.Where(type => !type.IsAbstract && !type.IsInterface))
        {
            TypeDefinition[] hierarchy = Hierarchy(type).ToArray();
            MethodDefinition[] facts = hierarchy.SelectMany(owner => owner.Methods)
                .Where(method => method.CustomAttributes.Any(attribute => Derives(attribute.AttributeType, "Xunit.FactAttribute"))).ToArray();
            if (facts.Length == 0) continue;
            var lifetime = new HashSet<string>(StringComparer.Ordinal);
            AddLifetime(type, lifetime);
            string classIdentity = assembly.Name.Name + ":" + type.FullName.Replace('/', '+');
            bool classComplete = !type.HasGenericParameters;
            foreach (TypeDefinition owner in hierarchy)
            {
                foreach (InterfaceImplementation implementation in owner.Interfaces)
                    if (implementation.InterfaceType is GenericInstanceType fixture && fixture.ElementType.FullName == "Xunit.IClassFixture`1")
                        AddFixture(fixture.GenericArguments[0], "class:" + classIdentity, lifetime, synthetic);
                foreach (CustomAttribute attribute in owner.CustomAttributes)
                {
                    AddBeforeAfter(attribute, lifetime);
                    if (attribute.AttributeType.FullName == "Xunit.TestCaseOrdererAttribute")
                        AddAll(NamedType(attribute, assembly), lifetime, "class-orderer:" + classIdentity);
                    if (attribute.AttributeType.FullName != "Xunit.CollectionAttribute") continue;
                    string? collection = attribute.ConstructorArguments.Count == 1 ? attribute.ConstructorArguments[0].Value switch
                    {
                        string name => name,
                        TypeReference reference => reference.FullName,
                        _ => null
                    } : null;
                    if (collection is null) { classComplete = false; continue; }
                    TypeDefinition[] definitions = types.Where(candidate => candidate.CustomAttributes.Any(definition =>
                        definition.AttributeType.FullName == "Xunit.CollectionDefinitionAttribute" &&
                        (definition.ConstructorArguments.Count == 0 ? candidate.FullName : definition.ConstructorArguments[0].Value as string) == collection)).ToArray();
                    if (definitions.Length > 1) { classComplete = false; continue; }
                    foreach (TypeDefinition definition in definitions)
                    {
                        foreach (TypeDefinition ancestor in Hierarchy(definition))
                            foreach (InterfaceImplementation implementation in ancestor.Interfaces)
                                if (implementation.InterfaceType is GenericInstanceType fixture)
                                {
                                    if (fixture.ElementType.FullName == "Xunit.ICollectionFixture`1")
                                        AddFixture(fixture.GenericArguments[0], "collection:" + assembly.Name.Name + ":" + collection, lifetime, synthetic);
                                    else if (fixture.ElementType.FullName == "Xunit.IClassFixture`1")
                                        AddFixture(fixture.GenericArguments[0], "class:" + classIdentity, lifetime, synthetic);
                                }
                        foreach (CustomAttribute orderer in definition.CustomAttributes.Where(attribute => attribute.AttributeType.FullName == "Xunit.TestCaseOrdererAttribute"))
                            AddAll(NamedType(orderer, assembly), lifetime, "collection-orderer:" + collection);
                    }
                }
            }
            foreach (IGrouping<string, MethodDefinition> overloads in facts.GroupBy(method => method.Name, StringComparer.Ordinal))
            {
                var roots = new HashSet<string>(lifetime, StringComparer.Ordinal);
                bool complete = classComplete;
                // Reflection inherits FactAttribute across overrides. Include
                // the concrete implementation even when it does not repeat the
                // attribute, plus base implementations reachable by base calls.
                foreach (MethodDefinition method in hierarchy.SelectMany(owner => owner.Methods).Where(method => method.Name == overloads.Key))
                {
                    roots.Add(Id(method));
                    complete &= !method.HasGenericParameters;
                    foreach (CustomAttribute attribute in method.CustomAttributes)
                    {
                        AddBeforeAfter(attribute, roots);
                        if (Derives(attribute.AttributeType, "Xunit.FactAttribute") &&
                            attribute.AttributeType.FullName is not ("Xunit.FactAttribute" or "Xunit.TheoryAttribute"))
                        {
                            complete = false;
                            AddAll(Resolve(attribute.AttributeType), groups, "custom-fact:" + attribute.AttributeType.FullName);
                            groups.Add("unresolved:xunit:custom-discoverer:" + attribute.AttributeType.FullName);
                        }
                        if (attribute.AttributeType.FullName == "Xunit.MemberDataAttribute") AddMemberData(attribute, type, deferredTheories ? roots : groups);
                        else if (attribute.AttributeType.FullName == "Xunit.ClassDataAttribute" && attribute.ConstructorArguments.Count == 1 &&
                                 attribute.ConstructorArguments[0].Value is TypeReference dataType)
                            AddAll(Resolve(dataType), deferredTheories ? roots : groups, "class-data:" + dataType.FullName);
                        else if (Derives(attribute.AttributeType, "Xunit.Sdk.DataAttribute") &&
                                 attribute.AttributeType.FullName is not ("Xunit.InlineDataAttribute" or "Xunit.MemberDataAttribute" or "Xunit.ClassDataAttribute"))
                            groups.Add("unresolved:xunit:custom-data:" + attribute.AttributeType.FullName);
                    }
                }
                tests.Add(new(classIdentity + "." + overloads.Key, roots.Order(StringComparer.Ordinal).ToArray(), complete));
            }
        }
        return new(new(1, tests.OrderBy(test => test.Owner, StringComparer.Ordinal).ToArray(), groups.Order(StringComparer.Ordinal).ToArray()),
            synthetic.Values.ToArray());
    }

    private static void AddMemberData(CustomAttribute attribute, TypeDefinition type, HashSet<string> roots)
    {
        string? name = attribute.ConstructorArguments.Count > 0 ? attribute.ConstructorArguments[0].Value as string : null;
        TypeDefinition? owner = type;
        foreach (CustomAttributeNamedArgument property in attribute.Properties)
            if (property.Name == "MemberType" && property.Argument.Value is TypeReference reference) owner = Resolve(reference);
        MethodDefinition[] providers = owner is null || name is null ? [] : Hierarchy(owner).SelectMany(candidate => candidate.Methods)
            .Where(method => method.IsStatic && (method.Name == name || method.Name == "get_" + name)).ToArray();
        // Static-field data providers and unresolved/ambiguous providers remain
        // open until their initialization and values are modeled explicitly.
        if (providers.Length != 1) roots.Add("unresolved:xunit:member-data:" + type.FullName + ":" + name);
        else roots.Add(Id(providers[0]));
    }

    private static void AddFixture(TypeReference reference, string scope, HashSet<string> roots, Dictionary<string, SourceMethod> synthetic)
    {
        TypeDefinition? fixture = Resolve(reference);
        string id = "xunit-fixture:" + scope + ":" + reference.FullName;
        var lifetime = new HashSet<string>(StringComparer.Ordinal);
        AddLifetime(fixture, lifetime);
        string hash = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(id)));
        synthetic.TryAdd(id, new(new(id, lifetime.Order(StringComparer.Ordinal).ToArray(), [id], DependencyBoundary.Closed), id, hash, [], false));
        roots.Add(id);
    }

    private static void AddBeforeAfter(CustomAttribute attribute, HashSet<string> roots)
    {
        if (!Derives(attribute.AttributeType, "Xunit.Sdk.BeforeAfterTestAttribute")) return;
        TypeDefinition? type = Resolve(attribute.AttributeType);
        AddLifetime(type, roots);
        // The runner invokes constructors, named setters and Before/After, not
        // every static helper inherited from System.Attribute. Calls made by
        // the hooks themselves remain ordinary transitive IL dependencies.
        HashSet<string> setters = attribute.Properties.Select(property => "set_" + property.Name).ToHashSet(StringComparer.Ordinal);
        foreach (TypeDefinition owner in Hierarchy(type))
            foreach (MethodDefinition method in owner.Methods.Where(method =>
                !method.IsStatic && (method.Name is "Before" or "After" || setters.Contains(method.Name))))
                roots.Add(Id(method));
    }

    private static void AddLifetime(TypeDefinition? type, HashSet<string> roots)
    {
        if (type is null) { roots.Add("unresolved:xunit:lifetime"); return; }
        foreach (TypeDefinition owner in Hierarchy(type))
            foreach (MethodDefinition method in owner.Methods.Where(method => method.IsConstructor ||
                method.Name.Split('.').Last() is "InitializeAsync" or "DisposeAsync" or "Dispose")) roots.Add(Id(method));
    }

    private static void AddFramework(TypeDefinition type, HashSet<string> roots)
    {
        foreach (TypeDefinition ancestor in Hierarchy(type))
        {
            if (ancestor.FullName == "AiDotNet.TestImpact.Xunit.AttributionTestFramework" && ancestor.Module.Assembly.Name.Name == "Attribution.Xunit" ||
                ancestor.FullName == "Xunit.Sdk.XunitTestFramework" && ancestor.Module.Assembly.Name.Name.StartsWith("xunit.execution.", StringComparison.Ordinal)) return;
            foreach (MethodDefinition method in ancestor.Methods.Where(method => method.HasBody)) roots.Add(Id(method));
        }
        roots.Add("unresolved:xunit:framework-base:" + type.FullName);
    }

    private static void AddAll(TypeDefinition? type, HashSet<string> roots, string missing)
    {
        if (type is null) { roots.Add("unresolved:xunit:" + missing); return; }
        foreach (TypeDefinition owner in Hierarchy(type))
            foreach (MethodDefinition method in owner.Methods.Where(method => method.HasBody)) roots.Add(Id(method));
    }

    private static TypeDefinition? NamedType(CustomAttribute attribute, AssemblyDefinition assembly)
    {
        if (attribute.ConstructorArguments.Count != 2 || attribute.ConstructorArguments[0].Value is not string type ||
            attribute.ConstructorArguments[1].Value is not string name) return null;
        if (name == assembly.Name.Name) return AllTypes(assembly.MainModule.Types).FirstOrDefault(candidate => candidate.FullName.Replace('/', '+') == type);
        try { return assembly.MainModule.AssemblyResolver.Resolve(new AssemblyNameReference(name, new Version(0, 0)))?.MainModule.GetType(type); }
        catch (AssemblyResolutionException) { return null; }
    }

    private static bool Derives(TypeReference reference, string name) => Hierarchy(Resolve(reference)).Any(type =>
        type.FullName == name && type.Module.Assembly.Name.Name == "xunit.core");
    private static TypeDefinition? Resolve(TypeReference reference)
    {
        if (reference is GenericParameter) return null;
        try { return reference.Resolve(); }
        catch (AssemblyResolutionException) { return null; }
        catch (ResolutionException) { return null; }
    }
    private static IEnumerable<TypeDefinition> Hierarchy(TypeDefinition? type)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        while (type is not null && type.FullName != "System.Object" && seen.Add(type.Module.Assembly.Name.Name + ":" + type.FullName))
        {
            yield return type;
            type = type.BaseType is null ? null : Resolve(type.BaseType);
        }
    }
    private static string Id(MethodDefinition method) => DependencyGraph.Stable(method);
    private static IEnumerable<TypeDefinition> AllTypes(IEnumerable<TypeDefinition> roots)
    {
        foreach (TypeDefinition type in roots)
        {
            yield return type;
            foreach (TypeDefinition nested in AllTypes(type.NestedTypes)) yield return nested;
        }
    }
}
