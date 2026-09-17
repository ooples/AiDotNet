using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;

internal enum OwnerConcurrencyContract { Unresolved, FixtureFreeNonparallelCollection }

// Conditional on the already verified standard xUnit runner. This declaration
// excludes concurrent xUnit collections, not arbitrary host threads. It is not
// a replacement for body/worker/observer proofs.
internal static class SerializedOwnerReader
{
    internal static OwnerConcurrencyContract Read(AssemblyDefinition assembly, string owner)
    {
        try
        {
            string prefix = assembly.Name.Name + ":";
            int separator = owner.LastIndexOf('.');
            if (!owner.StartsWith(prefix, StringComparison.Ordinal) || separator <= prefix.Length) return OwnerConcurrencyContract.Unresolved;
            TypeDefinition? type = assembly.MainModule.GetType(owner[prefix.Length..separator].Replace('+', '/'));
            if (type is null || type.IsAbstract || type.HasGenericParameters || type.IsValueType || type.HasInterfaces ||
                type.Fields.Any(field => !field.IsStatic || !field.IsLiteral) ||
                type.BaseType is null || !RuntimeType(type.BaseType, "System.Object") ||
                type.Methods.Count(method => method.Name == owner[(separator + 1)..]) != 1 ||
                !EmptyConstructor(type) || type.Methods.Any(method => method.Name == "Finalize") ||
                type.CustomAttributes.Any(attribute => attribute.AttributeType.FullName is
                    "Xunit.TestCaseOrdererAttribute" or "Xunit.TestCollectionOrdererAttribute")) return OwnerConcurrencyContract.Unresolved;
            // A custom collection factory can replace collection membership and
            // semantics. Only the ordinary assembly behavior constructor fits.
            CustomAttribute[] behavior = assembly.CustomAttributes.Where(attribute => attribute.AttributeType.FullName == "Xunit.CollectionBehaviorAttribute").ToArray();
            if (behavior.Length > 1 || behavior.Any(attribute => attribute.ConstructorArguments.Count != 0 ||
                    attribute.Constructor.Parameters.Count != 0 || !XunitAttribute(attribute, "Xunit.CollectionBehaviorAttribute")) ||
                assembly.CustomAttributes.Any(attribute => attribute.AttributeType.FullName is
                    "Xunit.TestCaseOrdererAttribute" or "Xunit.TestCollectionOrdererAttribute"))
                return OwnerConcurrencyContract.Unresolved;
            CustomAttribute[] memberships = type.CustomAttributes.Where(attribute => attribute.AttributeType.FullName == "Xunit.CollectionAttribute").ToArray();
            if (memberships.Length != 1 || !NamedAttribute(memberships[0], "Xunit.CollectionAttribute", out string name) ||
                memberships[0].Properties.Count != 0 || memberships[0].Fields.Count != 0)
                return OwnerConcurrencyContract.Unresolved;
            var definitions = Types(assembly.MainModule.Types).SelectMany(candidate => candidate.CustomAttributes
                .Where(attribute => attribute.AttributeType.FullName == "Xunit.CollectionDefinitionAttribute" &&
                    attribute.ConstructorArguments.Count == 1 && attribute.ConstructorArguments[0].Value is string value && value == name)
                .Select(attribute => (Type: candidate, Attribute: attribute))).ToArray();
            if (definitions.Length != 1) return OwnerConcurrencyContract.Unresolved;
            var definition = definitions[0];
            if (definition.Type.HasInterfaces || definition.Type.HasGenericParameters || definition.Type.BaseType is null ||
                !RuntimeType(definition.Type.BaseType, "System.Object") ||
                !NamedAttribute(definition.Attribute, "Xunit.CollectionDefinitionAttribute", out string declared) || declared != name ||
                definition.Attribute.Fields.Count != 0 || definition.Attribute.Properties.Count != 1 ||
                definition.Attribute.Properties[0].Name != nameof(Xunit.CollectionDefinitionAttribute.DisableParallelization) ||
                !RuntimeType(definition.Attribute.Properties[0].Argument.Type, "System.Boolean") ||
                definition.Attribute.Properties[0].Argument.Value is not true) return OwnerConcurrencyContract.Unresolved;
            return Hash(typeof(object).Assembly.Location) == ReviewedOwnerCompletion.RuntimeHash
                ? OwnerConcurrencyContract.FixtureFreeNonparallelCollection : OwnerConcurrencyContract.Unresolved;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or
            BadImageFormatException or AssemblyResolutionException or ResolutionException or InvalidOperationException or
            NotSupportedException or NotImplementedException)
        {
            return OwnerConcurrencyContract.Unresolved;
        }
    }

    private static bool EmptyConstructor(TypeDefinition type)
    {
        MethodDefinition[] constructors = type.Methods.Where(method => method.IsConstructor).ToArray();
        if (constructors.Length != 1) return false;
        MethodDefinition constructor = constructors[0];
        return !constructor.IsStatic && constructor.HasThis && constructor.Parameters.Count == 0 && constructor.HasBody && !constructor.HasGenericParameters &&
            constructor.ImplAttributes == MethodImplAttributes.IL && !constructor.IsPInvokeImpl && !constructor.HasSecurityDeclarations &&
            !constructor.Body.HasVariables && !constructor.Body.HasExceptionHandlers &&
            constructor.Body.Instructions.Select(instruction => instruction.OpCode.Code).SequenceEqual(new[] { Code.Ldarg_0, Code.Call, Code.Ret }) &&
            constructor.Body.Instructions[1].Operand is MethodReference call && call.HasThis && !call.ExplicitThis &&
            !call.HasGenericParameters && call.CallingConvention == MethodCallingConvention.Default &&
            call.FullName == "System.Void System.Object::.ctor()" && call.Resolve() is MethodDefinition original &&
            original.FullName == call.FullName && RuntimeType(call.DeclaringType, "System.Object") && RuntimeType(call.ReturnType, "System.Void");
    }

    private static bool NamedAttribute(CustomAttribute attribute, string type, out string name)
    {
        name = "";
        if (!XunitAttribute(attribute, type) || attribute.ConstructorArguments.Count != 1 ||
            attribute.Constructor.Parameters.Count != 1 || !RuntimeType(attribute.Constructor.Parameters[0].ParameterType, "System.String") ||
            !RuntimeType(attribute.ConstructorArguments[0].Type, "System.String") ||
            attribute.ConstructorArguments[0].Value is not string value || string.IsNullOrWhiteSpace(value)) return false;
        name = value;
        return true;
    }

    private static bool XunitAttribute(CustomAttribute attribute, string type) => attribute.AttributeType.FullName == type &&
        attribute.Constructor.HasThis && !attribute.Constructor.ExplicitThis && !attribute.Constructor.HasGenericParameters &&
        attribute.Constructor.CallingConvention == MethodCallingConvention.Default && RuntimeType(attribute.Constructor.ReturnType, "System.Void") &&
        attribute.AttributeType.Resolve() is TypeDefinition definition && definition.FullName == type &&
        attribute.Constructor.Resolve() is MethodDefinition constructor && constructor.DeclaringType == definition &&
        constructor.IsConstructor && !constructor.IsStatic && Hash(definition.Module.FileName) == ReviewedOwnerCompletion.CoreHash;

    private static bool RuntimeType(TypeReference reference, string name) => reference is not TypeSpecification && reference.FullName == name &&
        reference.Resolve() is TypeDefinition definition && definition.FullName == name &&
        string.Equals(Path.GetFullPath(definition.Module.FileName), Path.GetFullPath(typeof(object).Assembly.Location),
            OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
    private static string Hash(string path) { using var stream = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(stream)); }
    private static IEnumerable<TypeDefinition> Types(IEnumerable<TypeDefinition> types)
    {
        foreach (TypeDefinition type in types)
        {
            yield return type;
            foreach (TypeDefinition nested in Types(type.NestedTypes)) yield return nested;
        }
    }
}
