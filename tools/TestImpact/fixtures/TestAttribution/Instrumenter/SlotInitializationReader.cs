using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;

internal enum SlotInitializationContract { Unresolved, CallbackFreeAllocations }

// Bounded review of parameterless Object/AsyncLocal allocations. This says
// nothing about later slot writes, execution-context flow, or filesystem work.
// Unknown constructors and any additional initializer instructions stay open.
internal static class SlotInitializationReader
{
    internal static SlotInitializationContract Read(TypeDefinition type)
    {
        try
        {
            string runtime = typeof(object).Assembly.Location;
            using var stream = File.OpenRead(runtime);
            if (Convert.ToHexStringLower(SHA256.HashData(stream)) != ReviewedOwnerCompletion.RuntimeHash ||
                type.HasGenericParameters) return SlotInitializationContract.Unresolved;
            MethodDefinition[] initializers = type.Methods.Where(method => method.IsConstructor && method.IsStatic).ToArray();
            if (initializers.Length != 1) return SlotInitializationContract.Unresolved;
            MethodDefinition initializer = initializers[0];
            if (!initializer.HasBody || initializer.ImplAttributes != MethodImplAttributes.IL ||
                initializer.IsPInvokeImpl || initializer.HasSecurityDeclarations || initializer.HasGenericParameters ||
                initializer.HasOverrides || initializer.HasThis ||
                initializer.Parameters.Count != 0 || initializer.ReturnType.MetadataType != MetadataType.Void ||
                initializer.Body.HasExceptionHandlers || initializer.Body.HasVariables)
                return SlotInitializationContract.Unresolved;
            var instructions = initializer.Body.Instructions;
            if (instructions.Count < 3 || instructions.Count % 2 != 1 || instructions[^1].OpCode.Code != Code.Ret)
                return SlotInitializationContract.Unresolved;
            var fields = new HashSet<FieldDefinition>();
            for (int index = 0; index < instructions.Count - 1; index += 2)
            {
                if (instructions[index].OpCode.Code != Code.Newobj ||
                    instructions[index].Operand is not MethodReference constructor ||
                    instructions[index + 1].OpCode.Code != Code.Stsfld ||
                    instructions[index + 1].Operand is not FieldReference target ||
                    target.Resolve() is not FieldDefinition field || field.DeclaringType != type ||
                    !field.IsStatic || !field.IsInitOnly || !field.IsPrivate || !fields.Add(field) ||
                    !Constructor(constructor, runtime) || Identity(field.FieldType) != Identity(constructor.DeclaringType))
                    return SlotInitializationContract.Unresolved;
            }
            return SlotInitializationContract.CallbackFreeAllocations;
        }
        catch (Exception error) when (error is IOException or InvalidDataException or UnauthorizedAccessException or ArgumentException or
            BadImageFormatException or AssemblyResolutionException or ResolutionException or InvalidOperationException)
        {
            return SlotInitializationContract.Unresolved;
        }
    }

    private static bool Constructor(MethodReference reference, string runtime)
    {
        if (reference.Name != ".ctor" || !reference.HasThis || reference.ExplicitThis || reference.HasGenericParameters ||
            reference.CallingConvention != MethodCallingConvention.Default || reference.Parameters.Count != 0 ||
            reference.ReturnType.MetadataType != MetadataType.Void || reference.Resolve() is not MethodDefinition method ||
            !method.IsConstructor || method.IsStatic || method.Parameters.Count != 0 ||
            !string.Equals(Path.GetFullPath(method.Module.FileName), Path.GetFullPath(runtime),
                OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal)) return false;
        // Resolve against the reviewed bytes, not just a mutable Cecil object's
        // claimed filename or a same-named constructor in another assembly.
        using var pinned = AssemblyDefinition.ReadAssembly(runtime);
        if (pinned.MainModule.LookupToken(method.MetadataToken) is not MethodDefinition original ||
            original.FullName != method.FullName || !original.IsConstructor || original.IsStatic || original.Parameters.Count != 0)
            return false;
        if (reference.DeclaringType is GenericInstanceType local)
            return local.ElementType.FullName == "System.Threading.AsyncLocal`1" && local.GenericArguments.Count == 1 &&
                local.GenericArguments[0] is not (GenericParameter or TypeSpecification) &&
                local.GenericArguments[0].Resolve() is TypeDefinition argument && !argument.HasGenericParameters &&
                method.DeclaringType.FullName == "System.Threading.AsyncLocal`1";
        return reference.DeclaringType.FullName == "System.Object" && method.DeclaringType.FullName == "System.Object";
    }

    private static string Identity(TypeReference type)
    {
        if (type is GenericInstanceType generic)
            return Identity(generic.ElementType) + "<" + string.Join(",", generic.GenericArguments.Select(Identity)) + ">";
        if (type is TypeSpecification or GenericParameter) throw new InvalidDataException("Unsupported slot type.");
        TypeDefinition definition = type.Resolve() ?? throw new InvalidDataException("Unresolved slot type.");
        return definition.Module.Assembly.Name.FullName + ":" + definition.FullName;
    }
}
