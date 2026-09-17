using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;

internal enum LockedInitializationContract { Unresolved, ConstantInsertIfAbsent }
internal enum LockedInitializationRequirement
{
    PrefixEffects, PrivateDefaultMapInitialization, NonNullStringKey, SuccessfulOwner,
    NoExceptionObservers, NoOtherMapAccess, NoEscapedMapOrLock, ExclusiveLifetime
}
internal sealed record LockedInitializationAssessment(LockedInitializationContract Contract, int Start,
    string Map, string Lock, string Key, int Value, LockedInitializationRequirement[] Requirements);

// Recognizes only the locked tail, not the constructor prefix or its type
// initializer. Constant insert-if-absent operations commute, even for repeated
// keys, but no assertion may infer purity while another operation can read the
// map, replace its comparer, mutate it, or observe an exceptional allocation.
internal static class LockedInitializationReader
{
    internal static LockedInitializationAssessment Read(MethodDefinition method)
    {
        LockedInitializationAssessment Unknown() => new(LockedInitializationContract.Unresolved, 0, "", "", "", 0, []);
        try
        {
            if (!method.IsConstructor || method.IsStatic || !method.HasThis || method.ExplicitThis ||
                method.HasGenericParameters || method.ImplAttributes != MethodImplAttributes.IL ||
                method.IsPInvokeImpl || method.HasSecurityDeclarations || !method.HasBody ||
                method.Body.ExceptionHandlers.Count != 1 || method.Body.Instructions.Count < 24 ||
                Hash(typeof(object).Assembly.Location) != ReviewedOwnerCompletion.RuntimeHash) return Unknown();
            var all = method.Body.Instructions;
            int start = all.Count - 24;
            Instruction[] tail = all.Skip(start).ToArray();
            Code[] shape = [Code.Ldsfld, Code.Stloc, Code.Ldc_I4, Code.Stloc,
                Code.Ldloc, Code.Ldloca, Code.Call, Code.Ldsfld, Code.Ldarg_0, Code.Ldfld,
                Code.Callvirt, Code.Brtrue, Code.Ldsfld, Code.Ldarg_0, Code.Ldfld, Code.Ldc_I4,
                Code.Callvirt, Code.Leave, Code.Ldloc, Code.Brfalse, Code.Ldloc, Code.Call,
                Code.Endfinally, Code.Ret];
            if (!tail.Select(item => Normalize(item.OpCode.Code)).SequenceEqual(shape) || Constant(tail[2]) != 0 ||
                tail[11].Operand != tail[17] || tail[17].Operand != tail[23] || tail[19].Operand != tail[22]) return Unknown();
            ExceptionHandler handler = method.Body.ExceptionHandlers[0];
            if (handler.HandlerType != ExceptionHandlerType.Finally || handler.TryStart != tail[4] ||
                handler.TryEnd != tail[18] || handler.HandlerStart != tail[18] || handler.HandlerEnd != tail[23] ||
                handler.FilterStart is not null || handler.CatchType is not null) return Unknown();
            FieldDefinition? gate = Field(tail[0]), map = Field(tail[7]), key = Field(tail[9]);
            if (gate is null || map is null || key is null || gate == map || gate == key || map == key ||
                new[] { gate, map, key }.Any(field => field.DeclaringType != method.DeclaringType || !field.IsPrivate || !field.IsInitOnly ||
                    field.HasCustomAttributes) || !gate.IsStatic || !map.IsStatic || key.IsStatic ||
                !Runtime(gate.FieldType, "System.Object") || !Runtime(key.FieldType, "System.String") ||
                Field(tail[12]) != map || Field(tail[14]) != key ||
                map.FieldType is not GenericInstanceType dictionary || dictionary.GenericArguments.Count != 2 ||
                !Runtime(dictionary.ElementType, "System.Collections.Generic.Dictionary`2") ||
                !Runtime(dictionary.GenericArguments[0], "System.String") || !Runtime(dictionary.GenericArguments[1], "System.Int32")) return Unknown();
            int gateLocal = Local(tail[1]), takenLocal = Local(tail[3]);
            if (gateLocal < 0 || takenLocal < 0 || gateLocal == takenLocal || gateLocal >= method.Body.Variables.Count ||
                takenLocal >= method.Body.Variables.Count || !Runtime(method.Body.Variables[gateLocal].VariableType, "System.Object") ||
                !Runtime(method.Body.Variables[takenLocal].VariableType, "System.Boolean") ||
                new[] { 4, 20 }.Any(index => Local(tail[index]) != gateLocal) ||
                new[] { 5, 18 }.Any(index => Local(tail[index]) != takenLocal) ||
                !RuntimeCall(tail[6], "System.Void System.Threading.Monitor::Enter(System.Object,System.Boolean&)") ||
                !RuntimeCall(tail[21], "System.Void System.Threading.Monitor::Exit(System.Object)") ||
                !DictionaryCall(tail[10], dictionary, "ContainsKey", "System.Boolean", 1) ||
                !DictionaryCall(tail[16], dictionary, "set_Item", "System.Void", 2)) return Unknown();
            // Prefix control flow must enter through the lock setup. Otherwise
            // even a canonical-looking tail may use an unacquired lock/local.
            foreach (Instruction prefix in all.Take(start))
            {
                IEnumerable<Instruction> targets = prefix.Operand is Instruction one ? [one] :
                    prefix.Operand is Instruction[] many ? many : [];
                if (targets.Any(target => all.IndexOf(target) > start)) return Unknown();
            }
            return new(LockedInitializationContract.ConstantInsertIfAbsent, start,
                map.FullName, gate.FullName, key.FullName, Constant(tail[15]),
                Enum.GetValues<LockedInitializationRequirement>());
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or
            BadImageFormatException or AssemblyResolutionException or ResolutionException or InvalidOperationException)
        {
            return Unknown();
        }
    }

    private static bool DictionaryCall(Instruction instruction, GenericInstanceType dictionary, string name, string result, int arguments)
    {
        if (instruction.Operand is not MethodReference call || !call.HasThis || call.ExplicitThis || call.HasGenericParameters ||
            call.CallingConvention != MethodCallingConvention.Default || call.Name != name || !Runtime(call.ReturnType, result) ||
            call.Parameters.Count != arguments || call.DeclaringType is not GenericInstanceType owner ||
            owner.ElementType.Resolve() != dictionary.ElementType.Resolve() || owner.GenericArguments.Count != 2 ||
            !Runtime(owner.GenericArguments[0], "System.String") || !Runtime(owner.GenericArguments[1], "System.Int32") ||
            call.Resolve() is not MethodDefinition definition || definition.IsStatic || definition.Name != name ||
            definition.Parameters.Count != arguments) return false;
        for (int index = 0; index < arguments; index++)
            if (call.Parameters[index].ParameterType is not GenericParameter parameter || parameter.Type != GenericParameterType.Type ||
                parameter.Position != index || parameter.Owner is not TypeReference genericOwner || genericOwner.Resolve() != dictionary.ElementType.Resolve()) return false;
        return true;
    }

    private static bool RuntimeCall(Instruction instruction, string signature) => instruction.Operand is MethodReference call &&
        !call.HasThis && !call.ExplicitThis && !call.HasGenericParameters && call.CallingConvention == MethodCallingConvention.Default &&
        call.FullName == signature && call.Resolve() is MethodDefinition definition && definition.FullName == signature &&
        Runtime(definition.DeclaringType, "System.Threading.Monitor");

    private static bool Runtime(TypeReference reference, string name) => reference is not TypeSpecification && reference.FullName == name &&
        reference.Resolve() is TypeDefinition definition && definition.FullName == name &&
        string.Equals(Path.GetFullPath(definition.Module.FileName), Path.GetFullPath(typeof(object).Assembly.Location),
            OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
    private static FieldDefinition? Field(Instruction instruction) => instruction.Operand is FieldReference field ? field.Resolve() : null;
    private static string Hash(string path) { using var stream = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(stream)); }
    private static int Local(Instruction instruction) => instruction.Operand is VariableDefinition local ? local.Index : instruction.OpCode.Code switch
    {
        Code.Ldloc_0 or Code.Stloc_0 => 0, Code.Ldloc_1 or Code.Stloc_1 => 1,
        Code.Ldloc_2 or Code.Stloc_2 => 2, Code.Ldloc_3 or Code.Stloc_3 => 3, _ => -1
    };
    private static int Constant(Instruction instruction) => instruction.OpCode.Code switch
    {
        Code.Ldc_I4_M1 => -1, Code.Ldc_I4_0 => 0, Code.Ldc_I4_1 => 1, Code.Ldc_I4_2 => 2,
        Code.Ldc_I4_3 => 3, Code.Ldc_I4_4 => 4, Code.Ldc_I4_5 => 5, Code.Ldc_I4_6 => 6,
        Code.Ldc_I4_7 => 7, Code.Ldc_I4_8 => 8, Code.Ldc_I4_S => (sbyte)instruction.Operand,
        Code.Ldc_I4 => (int)instruction.Operand, _ => throw new InvalidOperationException()
    };
    private static Code Normalize(Code code) => code switch
    {
        Code.Ldloc_0 or Code.Ldloc_1 or Code.Ldloc_2 or Code.Ldloc_3 or Code.Ldloc_S => Code.Ldloc,
        Code.Stloc_0 or Code.Stloc_1 or Code.Stloc_2 or Code.Stloc_3 or Code.Stloc_S => Code.Stloc,
        Code.Ldloca_S => Code.Ldloca, Code.Brtrue_S => Code.Brtrue, Code.Brfalse_S => Code.Brfalse, Code.Leave_S => Code.Leave,
        Code.Ldc_I4_M1 or Code.Ldc_I4_0 or Code.Ldc_I4_1 or Code.Ldc_I4_2 or Code.Ldc_I4_3 or Code.Ldc_I4_4 or Code.Ldc_I4_5 or
            Code.Ldc_I4_6 or Code.Ldc_I4_7 or Code.Ldc_I4_8 or Code.Ldc_I4_S => Code.Ldc_I4,
        _ => code
    };
}
