using System.Security.Cryptography;
using AttributionRuntime;
using Mono.Cecil;
using Mono.Cecil.Cil;

internal enum TrialHookContract { Unresolved, ObservedSaveRestore }
internal enum TrialHookFailure { None, Shape, Binding, Scope, Cleanup, Construction, Resolution }
internal enum TrialHookRequirement { ObservedOwner, BodySlotIsolation, BodyFileIsolation, OwnerContextFlow }
internal sealed record TrialHookAssessment(TrialHookContract Contract, string Slot, TrialHookRequirement[] Requirements,
    TrialHookFailure Failure = TrialHookFailure.None);

// Proves the ordering and arguments of the opt-in Before/After observations.
// Cleanup is limited to the captured path and its tombstone. This still does
// not turn boundary samples into a proof that the test body performed no I/O.
internal static class TrialHookReader
{
    internal static TrialHookAssessment Read(MethodDefinition before, MethodDefinition after)
    {
        TrialHookAssessment Unknown(TrialHookFailure reason) => new(TrialHookContract.Unresolved, "", [], reason);
        try
        {
            TypeDefinition type = before.DeclaringType;
            if (after.DeclaringType != type || !type.IsSealed || type.HasGenericParameters || type.Fields.Count != 1 || type.Methods.Count != 5 ||
                !Callback(before, "Before") || !Callback(after, "After") ||
                SlotInitializationReader.Read(type) != SlotInitializationContract.CallbackFreeAllocations ||
                !Shape(before, Code.Call, Code.Ldstr, Code.Call, Code.Stloc_2, Code.Ldloca_S, Code.Ldstr, Code.Call,
                    Code.Ldstr, Code.Call, Code.Call, Code.Stloc_0, Code.Call, Code.Stloc_1, Code.Ldsfld, Code.Ldloc_0,
                    Code.Call, Code.Callvirt, Code.Ldloc_0, Code.Ldloc_1, Code.Call, Code.Call, Code.Ret) ||
                !Shape(after, Code.Call, Code.Stloc_0, Code.Ldsfld, Code.Callvirt, Code.Dup, Code.Brtrue_S, Code.Pop,
                    Code.Br_S, Code.Callvirt, Code.Ldsfld, Code.Ldnull, Code.Callvirt, Code.Ldloc_0, Code.Call, Code.Call,
                    Code.Ldloc_0, Code.Call, Code.Ldloc_0, Code.Brfalse_S, Code.Ldloc_0, Code.Ldstr, Code.Call,
                    Code.Br_S, Code.Ldnull, Code.Call, Code.Ret)) return Unknown(TrialHookFailure.Shape);
            if (!Construction(type)) return Unknown(TrialHookFailure.Construction);
            var first = before.Body.Instructions;
            var last = after.Body.Instructions;
            FieldDefinition slot = type.Fields[0];
            if (!slot.IsPrivate || !slot.IsStatic || !slot.IsInitOnly ||
                slot.FieldType is not GenericInstanceType local || local.GenericArguments.Count != 1 ||
                !RuntimeType(local.ElementType, "System.Threading.AsyncLocal`1") ||
                !RuntimeType(local.GenericArguments[0], "System.IDisposable") ||
                Field(first[13]) != slot || Field(last[2]) != slot || Field(last[9]) != slot ||
                !Locals(before, "System.String", "System.String", "System.Guid") || !Locals(after, "System.String") ||
                first[4].Operand != before.Body.Variables[2] ||
                !Literal(first[1], "aidotnet-trial-tests") || !Literal(first[5], "N") || !Literal(first[7], ".json") ||
                !RuntimeCall(first[0], "System.String System.IO.Path::GetTempPath()") ||
                !RuntimeCall(first[2], "System.Guid System.Guid::NewGuid()") ||
                !RuntimeCall(first[6], "System.String System.Guid::ToString(System.String)") ||
                !RuntimeCall(first[8], "System.String System.String::Concat(System.String,System.String)") ||
                !RuntimeCall(first[9], "System.String System.IO.Path::Combine(System.String,System.String,System.String)") ||
                !Accessor(first[16], local, true) || !Accessor(last[3], local, false) || !Accessor(last[11], local, true) ||
                !RuntimeCall(last[8], "System.Void System.IDisposable::Dispose()") ||
                !Branch(after, 5, 8) || !Branch(after, 7, 9) || !Branch(after, 18, 23) || !Branch(after, 22, 24) ||
                !Literal(last[20], ".tombstone") ||
                !RuntimeCall(last[21], "System.String System.String::Concat(System.String,System.String)") ||
                !Observer(first[20], nameof(Tracker.TrialScopeStarted), 3) ||
                !Observer(last[14], nameof(Tracker.TrialScopeEnded), 2)) return Unknown(TrialHookFailure.Binding);
            MethodDefinition? factory = Method(first[15]), getter = Method(first[11]), cleanup = Method(last[16]);
            if (factory is null || getter is null || Method(first[19]) != getter || Method(last[0]) != getter || Method(last[13]) != getter ||
                cleanup is null || Method(last[24]) != cleanup || cleanup.DeclaringType != type || !cleanup.IsPrivate || !cleanup.IsStatic ||
                cleanup.HasGenericParameters || cleanup.Parameters.Count != 1 || !RuntimeType(cleanup.Parameters[0].ParameterType, "System.String") ||
                !RuntimeType(cleanup.ReturnType, "System.Void")) return Unknown(TrialHookFailure.Binding);
            if (!Cleanup(cleanup)) return Unknown(TrialHookFailure.Cleanup);
            AsyncLocalScopeAssessment scope = ScopedAsyncLocalReader.Read(factory, getter);
            if (scope.Contract != AsyncLocalScopeContract.RestoresPreviousString ||
                scope.Initialization != SlotInitializationContract.CallbackFreeAllocations) return Unknown(TrialHookFailure.Scope);
            return new(TrialHookContract.ObservedSaveRestore, scope.Slot, Enum.GetValues<TrialHookRequirement>());
        }
        catch (Exception error) when (error is IOException or InvalidDataException or UnauthorizedAccessException or ArgumentException or
            BadImageFormatException or AssemblyResolutionException or ResolutionException or InvalidOperationException)
        {
            return Unknown(TrialHookFailure.Resolution);
        }
    }

    private static bool Construction(TypeDefinition type)
    {
        MethodDefinition[] constructors = type.Methods.Where(method => method.IsConstructor && !method.IsStatic).ToArray();
        if (constructors.Length != 1 || constructors[0].Parameters.Count != 0 || !constructors[0].HasThis || constructors[0].HasGenericParameters ||
            !RuntimeType(constructors[0].ReturnType, "System.Void") ||
            !Shape(constructors[0], Code.Ldarg_0, Code.Call, Code.Ret) || constructors[0].Body.HasVariables ||
            constructors[0].Body.Instructions[1].Operand is not MethodReference reference ||
            reference.FullName != "System.Void Xunit.Sdk.BeforeAfterTestAttribute::.ctor()" ||
            !reference.HasThis || reference.ExplicitThis || reference.HasGenericParameters ||
            reference.CallingConvention != MethodCallingConvention.Default || reference.Resolve() is not MethodDefinition baseConstructor ||
            type.BaseType is not TypeReference baseType || baseConstructor.DeclaringType != baseType.Resolve()) return false;
        using var stream = File.OpenRead(baseConstructor.Module.FileName);
        return Convert.ToHexStringLower(SHA256.HashData(stream)) == ReviewedOwnerCompletion.CoreHash;
    }

    private static bool Cleanup(MethodDefinition method)
    {
        if (!SafeBody(method) || !Locals(method, "System.Exception", "System.Boolean") || method.Body.ExceptionHandlers.Count != 1 ||
            !method.Body.Instructions.Select(instruction => instruction.OpCode.Code).SequenceEqual(new[]
            {
                Code.Ldarg_0, Code.Call, Code.Brfalse_S, Code.Ret, Code.Nop, Code.Ldarg_0, Code.Call, Code.Brfalse_S,
                Code.Ldarg_0, Code.Call, Code.Leave_S, Code.Isinst, Code.Dup, Code.Brtrue_S, Code.Pop, Code.Ldc_I4_0,
                Code.Br_S, Code.Stloc_0, Code.Ldloc_0, Code.Isinst, Code.Brtrue_S, Code.Ldloc_0, Code.Isinst,
                Code.Brfalse_S, Code.Ldc_I4_1, Code.Stloc_1, Code.Br_S, Code.Ldc_I4_0, Code.Stloc_1, Code.Ldloc_1,
                Code.Ldc_I4_0, Code.Cgt_Un, Code.Endfilter, Code.Pop, Code.Leave_S, Code.Ret
            })) return false;
        var instructions = method.Body.Instructions;
        ExceptionHandler handler = method.Body.ExceptionHandlers[0];
        return handler.HandlerType == ExceptionHandlerType.Filter && handler.CatchType is null &&
            handler.TryStart == instructions[5] && handler.TryEnd == instructions[11] && handler.FilterStart == instructions[11] &&
            handler.HandlerStart == instructions[33] && handler.HandlerEnd == instructions[35] &&
            Branch(method, 2, 4) && Branch(method, 7, 10) && Branch(method, 10, 35) && Branch(method, 13, 17) &&
            Branch(method, 16, 32) && Branch(method, 20, 24) && Branch(method, 23, 27) && Branch(method, 26, 29) && Branch(method, 34, 35) &&
            RuntimeCall(instructions[1], "System.Boolean System.String::IsNullOrEmpty(System.String)") &&
            RuntimeCall(instructions[6], "System.Boolean System.IO.File::Exists(System.String)") &&
            RuntimeCall(instructions[9], "System.Void System.IO.File::Delete(System.String)") &&
            instructions[11].Operand is TypeReference exception && RuntimeType(exception, "System.Exception") &&
            instructions[19].Operand is TypeReference io && RuntimeType(io, "System.IO.IOException") &&
            instructions[22].Operand is TypeReference access && RuntimeType(access, "System.UnauthorizedAccessException");
    }

    private static bool Callback(MethodDefinition method, string name) => method.Name == name && method.IsPublic && method.IsVirtual &&
        !method.IsNewSlot && !method.IsStatic && !method.HasOverrides && !method.HasGenericParameters && method.Parameters.Count == 1 &&
        RuntimeType(method.Parameters[0].ParameterType, "System.Reflection.MethodInfo") && RuntimeType(method.ReturnType, "System.Void");

    private static bool SafeBody(MethodDefinition method) => method.HasBody &&
        method.ImplAttributes == MethodImplAttributes.IL && !method.IsPInvokeImpl && !method.HasSecurityDeclarations;
    private static bool Shape(MethodDefinition method, params Code[] codes) => SafeBody(method) &&
        !method.Body.HasExceptionHandlers && method.Body.Instructions.Select(instruction => instruction.OpCode.Code).SequenceEqual(codes);

    private static bool Locals(MethodDefinition method, params string[] types) => method.Body.Variables.Count == types.Length &&
        types.Select((name, index) => RuntimeType(method.Body.Variables[index].VariableType, name)).All(value => value);
    private static bool Literal(Instruction instruction, string expected) => instruction.Operand is string value && value == expected;
    private static bool Branch(MethodDefinition method, int from, int to) => method.Body.Instructions[from].Operand == method.Body.Instructions[to];
    private static FieldDefinition? Field(Instruction instruction) => instruction.Operand is FieldReference field ? field.Resolve() : null;
    private static MethodDefinition? Method(Instruction instruction) => instruction.Operand is MethodReference method ? method.Resolve() : null;
    private static bool RuntimeType(TypeReference type, string name) => type.FullName == name && type.Resolve() is TypeDefinition definition &&
        definition.FullName == name && SameFile(definition.Module.FileName, typeof(object).Assembly.Location);
    private static bool RuntimeCall(Instruction instruction, string signature) => instruction.Operand is MethodReference reference &&
        !reference.ExplicitThis && !reference.HasGenericParameters && reference.CallingConvention == MethodCallingConvention.Default &&
        reference.FullName == signature && reference.Resolve() is MethodDefinition method && method.FullName == signature &&
        reference.HasThis == method.HasThis && RuntimeType(reference.ReturnType, method.ReturnType.FullName) &&
        reference.Parameters.All(parameter => RuntimeType(parameter.ParameterType, parameter.ParameterType.FullName)) &&
        SameFile(method.Module.FileName, typeof(object).Assembly.Location);

    private static bool Accessor(Instruction instruction, GenericInstanceType local, bool setter) =>
        instruction.Operand is MethodReference reference && reference.HasThis && !reference.HasGenericParameters && !reference.ExplicitThis &&
        reference.CallingConvention == MethodCallingConvention.Default && reference.DeclaringType is GenericInstanceType instance &&
        instance.GenericArguments.Count == 1 && RuntimeType(instance.GenericArguments[0], "System.IDisposable") &&
        reference.DeclaringType.FullName == local.FullName && reference.Name == (setter ? "set_Value" : "get_Value") &&
        reference.Parameters.Count == (setter ? 1 : 0) && reference.Resolve() is MethodDefinition definition &&
        definition.DeclaringType.FullName == "System.Threading.AsyncLocal`1" &&
        SameFile(definition.Module.FileName, typeof(object).Assembly.Location) &&
        (setter ? RuntimeType(reference.ReturnType, "System.Void") && Parameter(reference.Parameters[0].ParameterType, instance)
            : Parameter(reference.ReturnType, instance));

    private static bool Parameter(TypeReference type, GenericInstanceType instance) => type is GenericParameter parameter &&
        parameter.Type == GenericParameterType.Type && parameter.Position == 0 &&
        parameter.Owner is TypeReference owner && owner.Resolve() == instance.ElementType.Resolve();

    private static bool Observer(Instruction instruction, string name, int count)
    {
        if (instruction.Operand is not MethodReference reference || reference.HasThis || reference.HasGenericParameters || reference.ExplicitThis ||
            reference.CallingConvention != MethodCallingConvention.Default ||
            reference.Name != name || reference.DeclaringType.FullName != typeof(Tracker).FullName || reference.Parameters.Count != count ||
            !RuntimeType(reference.ReturnType, "System.Void") || reference.Parameters.Any(parameter => !RuntimeType(parameter.ParameterType, "System.String")) ||
            reference.Resolve() is not MethodDefinition method || method.FullName != reference.FullName) return false;
        using var bound = File.OpenRead(method.Module.FileName);
        using var trusted = File.OpenRead(typeof(Tracker).Assembly.Location);
        return SHA256.HashData(bound).SequenceEqual(SHA256.HashData(trusted));
    }

    private static bool SameFile(string first, string second) => string.Equals(Path.GetFullPath(first), Path.GetFullPath(second),
        OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
}
