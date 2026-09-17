using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;

internal enum AsyncLocalScopeContract { Unresolved, RestoresPreviousString }
internal enum AsyncLocalScopeRequirement { Initializer, ExclusiveLifetime, NoExternalSlotMutation, OwnerContextFlow }
internal sealed record AsyncLocalScopeAssessment(AsyncLocalScopeContract Contract, string Slot, string RuntimeHash,
    AsyncLocalScopeRequirement[] Requirements, SlotInitializationContract Initialization = SlotInitializationContract.Unresolved);

// Proves the bounded save/set/restore operation, not arbitrary AsyncLocal use
// or file isolation. In particular, a body can still replace the slot or dispose
// a scope twice. Those lifetime/ownership requirements cannot be inferred from
// a passing outer test or from the name of the persistence helper.
internal static class ScopedAsyncLocalReader
{
    internal static AsyncLocalScopeAssessment Read(MethodDefinition factory, MethodDefinition getter)
    {
        AsyncLocalScopeAssessment Unknown() => new(AsyncLocalScopeContract.Unresolved, "", "", []);
        try
        {
            string runtime = typeof(object).Assembly.Location;
            if (Hash(runtime) != ReviewedOwnerCompletion.RuntimeHash || !factory.IsStatic || !getter.IsStatic ||
                factory.HasGenericParameters || getter.HasGenericParameters || factory.Parameters.Count != 1 || getter.Parameters.Count != 0 ||
                !RuntimeType(factory.Parameters[0].ParameterType, "System.String", runtime) ||
                !RuntimeType(factory.ReturnType, "System.IDisposable", runtime) ||
                !RuntimeType(getter.ReturnType, "System.String", runtime) ||
                !FactoryShape(factory, runtime, out int secondSlot, out int allocation) ||
                !Shape(getter, Code.Ldsfld, Code.Callvirt, Code.Ret)) return Unknown();
            var body = factory.Body.Instructions;
            FieldDefinition? slot = Field(body[0]);
            if (slot is null || !slot.IsStatic || !slot.IsInitOnly || slot.DeclaringType != factory.DeclaringType ||
                slot != Field(body[secondSlot]) || slot != Field(getter.Body.Instructions[0]) ||
                slot.FieldType is not GenericInstanceType local || local.GenericArguments.Count != 1 ||
                !RuntimeType(local.ElementType, "System.Threading.AsyncLocal`1", runtime) ||
                !RuntimeType(local.GenericArguments[0], "System.String", runtime) ||
                !Accessor(body[1], local, false, runtime) || !Accessor(body[secondSlot + 2], local, true, runtime) ||
                !Accessor(getter.Body.Instructions[1], local, false, runtime) ||
                body[allocation].Operand is not MethodReference allocated || allocated.Resolve() is not MethodDefinition constructor)
                return Unknown();
            TypeDefinition scope = constructor.DeclaringType;
            MethodDefinition[] disposers = scope.Methods.Where(method => method.Name == "Dispose").ToArray();
            if (!scope.IsSealed || scope.IsValueType || scope.HasGenericParameters || scope.Fields.Count != 1 ||
                scope.Methods.Count != 2 || disposers.Length != 1 || !constructor.IsConstructor || constructor.IsStatic ||
                constructor.Parameters.Count != 1 || !RuntimeType(constructor.Parameters[0].ParameterType, "System.String", runtime) ||
                !RuntimeType(scope.BaseType, "System.Object", runtime) || scope.Interfaces.Count != 1 ||
                !RuntimeType(scope.Interfaces[0].InterfaceType, "System.IDisposable", runtime) ||
                !Shape(constructor, Code.Ldarg_0, Code.Call, Code.Ldarg_0, Code.Ldarg_1, Code.Stfld, Code.Ret)) return Unknown();
            FieldDefinition previous = scope.Fields[0];
            MethodDefinition dispose = disposers[0];
            if (previous.IsStatic || !previous.IsInitOnly || !previous.IsPrivate ||
                !RuntimeType(previous.FieldType, "System.String", runtime) || Field(constructor.Body.Instructions[4]) != previous ||
                !ObjectConstructor(constructor.Body.Instructions[1], runtime) || dispose.IsStatic || !dispose.IsPublic ||
                !dispose.IsVirtual || !dispose.IsFinal || dispose.HasOverrides || dispose.HasGenericParameters || dispose.Parameters.Count != 0 ||
                !RuntimeType(dispose.ReturnType, "System.Void", runtime) ||
                !Shape(dispose, Code.Ldsfld, Code.Ldarg_0, Code.Ldfld, Code.Callvirt, Code.Ret) ||
                Field(dispose.Body.Instructions[0]) != slot || Field(dispose.Body.Instructions[2]) != previous ||
                !Accessor(dispose.Body.Instructions[3], local, true, runtime)) return Unknown();
            // The exact parameterless initializer has no change-notification
            // callback. Other work in the declaring cctor remains a requirement.
            MethodDefinition[] initializers = slot.DeclaringType.Methods.Where(method => method.IsConstructor && method.IsStatic).ToArray();
            if (initializers.Length != 1 || !initializers[0].HasBody) return Unknown();
            var instructions = initializers[0].Body.Instructions;
            int[] writes = instructions.Select((instruction, index) => (instruction, index))
                .Where(item => item.instruction.OpCode.Code == Code.Stsfld && Field(item.instruction) == slot)
                .Select(item => item.index).ToArray();
            if (writes.Length != 1 || writes[0] == 0 || instructions[writes[0] - 1].OpCode.Code != Code.Newobj ||
                instructions[writes[0] - 1].Operand is not MethodReference creation || creation.Name != ".ctor" ||
                creation.Parameters.Count != 0 || !creation.HasThis || creation.DeclaringType.FullName != local.FullName ||
                !RuntimeMethod(creation, runtime) || Hash(runtime) != ReviewedOwnerCompletion.RuntimeHash) return Unknown();
            SlotInitializationContract initialization = SlotInitializationReader.Read(slot.DeclaringType);
            AsyncLocalScopeRequirement[] requirements = Enum.GetValues<AsyncLocalScopeRequirement>()
                .Where(requirement => requirement != AsyncLocalScopeRequirement.Initializer ||
                    initialization == SlotInitializationContract.Unresolved).ToArray();
            return new(AsyncLocalScopeContract.RestoresPreviousString, slot.Module.Assembly.Name.Name + ":" + slot.FullName,
                ReviewedOwnerCompletion.RuntimeHash, requirements, initialization);
        }
        catch (Exception error) when (error is IOException or InvalidDataException or UnauthorizedAccessException or ArgumentException or
            BadImageFormatException or AssemblyResolutionException or ResolutionException or InvalidOperationException)
        {
            return Unknown();
        }
    }

    private static bool FactoryShape(MethodDefinition method, string runtime, out int secondSlot, out int allocation)
    {
        secondSlot = 2; allocation = 5;
        if (Shape(method, Code.Ldsfld, Code.Callvirt, Code.Ldsfld, Code.Ldarg_0, Code.Callvirt, Code.Newobj, Code.Ret)) return true;
        // The non-stack form must save and reload precisely one string local.
        // Do not drop arbitrary local traffic or normalize another slot to zero.
        secondSlot = 3; allocation = 7;
        return SafeBody(method) && method.Body.ExceptionHandlers.Count == 0 && method.Body.Variables.Count == 1 &&
            RuntimeType(method.Body.Variables[0].VariableType, "System.String", runtime) &&
            method.Body.Instructions.Select(instruction => instruction.OpCode.Code).SequenceEqual(new[]
            { Code.Ldsfld, Code.Callvirt, Code.Stloc_0, Code.Ldsfld, Code.Ldarg_0, Code.Callvirt, Code.Ldloc_0, Code.Newobj, Code.Ret });
    }

    private static bool SafeBody(MethodDefinition method) => method.HasBody && method.ImplAttributes == MethodImplAttributes.IL &&
        !method.IsPInvokeImpl && !method.HasSecurityDeclarations;

    private static bool Shape(MethodDefinition method, params Code[] codes) => SafeBody(method) &&
        method.Body.ExceptionHandlers.Count == 0 && method.Body.Variables.Count == 0 &&
        method.Body.Instructions.Select(instruction => instruction.OpCode.Code).SequenceEqual(codes);

    private static FieldDefinition? Field(Instruction instruction) => instruction.Operand is FieldReference field ? field.Resolve() : null;

    private static bool Accessor(Instruction instruction, GenericInstanceType local, bool setter, string runtime) =>
        instruction.Operand is MethodReference method && method.HasThis && !method.HasGenericParameters &&
        method.DeclaringType.FullName == local.FullName && method.Name == (setter ? "set_Value" : "get_Value") &&
        method.Parameters.Count == (setter ? 1 : 0) && RuntimeMethod(method, runtime) &&
        (setter ? method.ReturnType.MetadataType == MetadataType.Void && Parameter(method.Parameters[0].ParameterType)
            : Parameter(method.ReturnType));

    private static bool Parameter(TypeReference type) => type is GenericParameter parameter &&
        parameter.Type == GenericParameterType.Type && parameter.Position == 0;

    private static bool ObjectConstructor(Instruction instruction, string runtime) => instruction.Operand is MethodReference method &&
        method.Name == ".ctor" && method.HasThis && method.Parameters.Count == 0 &&
        RuntimeType(method.DeclaringType, "System.Object", runtime) && RuntimeMethod(method, runtime);

    private static bool RuntimeType(TypeReference? type, string name, string runtime) => type is not null &&
        type.FullName == name && type.Resolve() is TypeDefinition definition && definition.FullName == name && SamePath(definition.Module.FileName, runtime);

    private static bool RuntimeMethod(MethodReference method, string runtime) => method.Resolve() is MethodDefinition definition &&
        SamePath(definition.Module.FileName, runtime) && definition.Name == method.Name;

    private static bool SamePath(string left, string right) => string.Equals(Path.GetFullPath(left), Path.GetFullPath(right),
        OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);

    private static string Hash(string path)
    {
        for (FileSystemInfo? entry = new FileInfo(Path.GetFullPath(path)); entry is not null;
             entry = entry is FileInfo file ? file.Directory : ((DirectoryInfo)entry).Parent)
            if (!entry.Exists) throw new FileNotFoundException("Missing scope contract input.", entry.FullName);
            else if ((entry.Attributes & FileAttributes.ReparsePoint) != 0 || entry.LinkTarget is not null)
                throw new InvalidDataException("Linked scope contract input.");
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }
}
