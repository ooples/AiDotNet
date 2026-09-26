using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;

internal enum PrivateMapInitializerContract { Unresolved, FreshPrivateDefaultStringMaps }
internal sealed record PrivateMapInitializerAssessment(PrivateMapInitializerContract Contract, string[] Fields);

// Only describes the initial storage. It does not prove that later operations
// cannot escape or observe it; that remains a whole-workload obligation.
internal static class PrivateMapInitializerReader
{
    internal static PrivateMapInitializerAssessment Read(TypeDefinition owner)
    {
        PrivateMapInitializerAssessment Unknown() => new(PrivateMapInitializerContract.Unresolved, []);
        try
        {
            if (Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(object).Assembly.Location))) != ReviewedOwnerCompletion.RuntimeHash)
                return Unknown();
            MethodDefinition[] initializers = owner.Methods.Where(method => method.IsConstructor && method.IsStatic).ToArray();
            if (initializers.Length != 1) return Unknown();
            MethodDefinition initializer = initializers[0];
            if (!initializer.HasBody || initializer.HasThis || initializer.ExplicitThis || initializer.HasGenericParameters ||
                initializer.Parameters.Count != 0 || initializer.ImplAttributes != MethodImplAttributes.IL || initializer.IsPInvokeImpl ||
                initializer.HasSecurityDeclarations || initializer.Body.HasExceptionHandlers || initializer.Body.HasVariables ||
                !Runtime(initializer.ReturnType, "System.Void")) return Unknown();
            var il = initializer.Body.Instructions;
            if (il.Count < 5 || il.Count % 2 != 1 || il[^1].OpCode.Code != Code.Ret) return Unknown();
            var written = new HashSet<FieldDefinition>();
            int maps = 0, gates = 0;
            for (int index = 0; index < il.Count - 1; index += 2)
            {
                if (il[index].OpCode.Code != Code.Newobj || il[index + 1].OpCode.Code != Code.Stsfld ||
                    il[index].Operand is not MethodReference constructor || constructor.Name != ".ctor" || !constructor.HasThis ||
                    constructor.ExplicitThis || constructor.HasGenericParameters || constructor.CallingConvention != MethodCallingConvention.Default ||
                    constructor.Parameters.Count != 0 || !Runtime(constructor.ReturnType, "System.Void") ||
                    constructor.Resolve() is not MethodDefinition definition || !definition.IsConstructor || definition.IsStatic ||
                    OwnedFieldBinding.Read(il[index + 1], owner) is not FieldDefinition field || !field.IsStatic || !field.IsPrivate ||
                    !field.IsInitOnly || field.CustomAttributes.Any(attribute => !Runtime(attribute.AttributeType, "System.Runtime.CompilerServices.NullableAttribute")) ||
                    field.HasConstant || field.InitialValue.Length != 0 || !written.Add(field) ||
                    !OwnedFieldBinding.SameType(field.FieldType, constructor.DeclaringType)) return Unknown();
                if (Runtime(constructor.DeclaringType, "System.Object")) { gates++; continue; }
                if (constructor.DeclaringType is not GenericInstanceType dictionary || dictionary.GenericArguments.Count != 2 ||
                    !Runtime(dictionary.ElementType, "System.Collections.Generic.Dictionary`2") ||
                    !Runtime(dictionary.GenericArguments[0], "System.String")) return Unknown();
                // Pinned Dictionary<string,TValue>() delegates with capacity=0,
                // comparer=null. It initializes string comparers but never calls
                // TValue code or constructs a TValue. No caller-supplied comparer.
                maps++;
            }
            FieldDefinition[] storage = owner.Fields.Where(field => field.IsStatic && !field.IsLiteral).ToArray();
            return maps > 0 && gates == 1 && storage.Length == written.Count && storage.All(written.Contains)
                ? new(PrivateMapInitializerContract.FreshPrivateDefaultStringMaps, written.Select(field => field.FullName).Order(StringComparer.Ordinal).ToArray())
                : Unknown();
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or BadImageFormatException or
            AssemblyResolutionException or ResolutionException or InvalidOperationException)
        { return Unknown(); }
    }

    private static bool Runtime(TypeReference reference, string name) => reference is not TypeSpecification && reference.FullName == name &&
        reference.Resolve() is TypeDefinition definition && definition.FullName == name &&
        string.Equals(Path.GetFullPath(definition.Module.FileName), Path.GetFullPath(typeof(object).Assembly.Location),
            OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
}
