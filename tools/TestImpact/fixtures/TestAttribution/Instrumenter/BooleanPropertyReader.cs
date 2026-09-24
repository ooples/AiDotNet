using Mono.Cecil;
using Mono.Cecil.Cil;

internal enum BooleanAccessorKind { Read, Write }

internal static class BooleanPropertyReader
{
    internal static FieldDefinition? Read(MethodReference reference, TypeReference allocation, BooleanAccessorKind kind)
    {
        if (allocation is not GenericInstanceType expected || reference.DeclaringType is not GenericInstanceType actual ||
            expected.GenericArguments.Count != 1 || actual.GenericArguments.Count != 1 ||
            !Runtime(expected.GenericArguments[0], "System.Double") || !Runtime(actual.GenericArguments[0], "System.Double") ||
            expected.ElementType.Resolve() is not TypeDefinition owner || actual.ElementType.Resolve() != owner) return null;
        return Shape(reference, owner, kind);
    }

    internal static FieldDefinition? ReadSelf(MethodReference reference, TypeDefinition owner, BooleanAccessorKind kind) =>
        OwnedFieldBinding.SelfType(reference.DeclaringType, owner) ? Shape(reference, owner, kind) : null;

    private static FieldDefinition? Shape(MethodReference reference, TypeDefinition owner, BooleanAccessorKind kind)
    {
        if (!Enum.IsDefined(kind) || !reference.HasThis || reference.ExplicitThis || reference.HasGenericParameters ||
            reference.CallingConvention != MethodCallingConvention.Default || reference.Resolve() is not MethodDefinition method ||
            method.DeclaringType != owner || method.IsStatic || !method.HasBody || method.ImplAttributes != MethodImplAttributes.IL ||
            method.IsPInvokeImpl || method.HasSecurityDeclarations || method.HasGenericParameters ||
            method.Body.HasExceptionHandlers || method.Body.HasVariables) return null;
        var il = method.Body.Instructions;
        bool read = kind == BooleanAccessorKind.Read;
        Code[] codes = read ? [Code.Ldarg_0, Code.Ldfld, Code.Ret] : [Code.Ldarg_0, Code.Ldarg_1, Code.Stfld, Code.Ret];
        if (!il.Select(instruction => instruction.OpCode.Code).SequenceEqual(codes) ||
            reference.Parameters.Count != (read ? 0 : 1) || method.Parameters.Count != reference.Parameters.Count ||
            !Runtime(reference.ReturnType, read ? "System.Boolean" : "System.Void") ||
            !Runtime(method.ReturnType, read ? "System.Boolean" : "System.Void") ||
            !read && (!Runtime(reference.Parameters[0].ParameterType, "System.Boolean") || !Runtime(method.Parameters[0].ParameterType, "System.Boolean")) ||
            OwnedFieldBinding.Read(il[read ? 1 : 2], owner) is not FieldDefinition field || field.IsStatic || !field.IsPrivate ||
            !Runtime(field.FieldType, "System.Boolean")) return null;
        return field;
    }

    private static bool Runtime(TypeReference reference, string name) => reference is not TypeSpecification && reference.FullName == name &&
        reference.Resolve() is TypeDefinition definition && definition.FullName == name &&
        string.Equals(Path.GetFullPath(definition.Module.FileName), Path.GetFullPath(typeof(object).Assembly.Location),
            OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
}
