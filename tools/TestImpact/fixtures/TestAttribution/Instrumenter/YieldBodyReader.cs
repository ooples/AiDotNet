using System.Security.Cryptography;
using AiDotNet.TestImpact;
using Mono.Cecil;
using Mono.Cecil.Cil;

internal enum YieldBodyContract { Unresolved, SingleYieldTaskBody }
internal sealed record YieldBodyWindow(YieldBodyContract Contract, int Start, int Length);

// Identifies the entire user-body window inside one canonical awaited Yield.
// It does NOT declare that window isolated: every instruction/call inside it
// still needs an effect proof. Extra awaits, task substitution, capture fields,
// handlers, or prefix/suffix behavior must not hide outside that window.
internal static class YieldBodyReader
{
    internal static YieldBodyWindow ReadOwner(AssemblyDefinition assembly, string owner)
    {
        try
        {
            string prefix = assembly.Name.Name + ":";
            int separator = owner.LastIndexOf('.');
            if (!owner.StartsWith(prefix, StringComparison.Ordinal) || separator <= prefix.Length) return new(YieldBodyContract.Unresolved, 0, 0);
            TypeDefinition? type = assembly.MainModule.GetType(owner[prefix.Length..separator].Replace('+', '/'));
            MethodDefinition[] entries = type?.Methods.Where(method => method.Name == owner[(separator + 1)..]).ToArray() ?? [];
            if (entries.Length != 1) return new(YieldBodyContract.Unresolved, 0, 0);
            CustomAttribute[] markers = entries[0].CustomAttributes.Where(attribute => attribute.AttributeType.FullName ==
                "System.Runtime.CompilerServices.AsyncStateMachineAttribute").ToArray();
            if (markers.Length != 1 || markers[0].ConstructorArguments.Count != 1 ||
                markers[0].ConstructorArguments[0].Value is not TypeReference state || state.Resolve() is not TypeDefinition definition)
                return new(YieldBodyContract.Unresolved, 0, 0);
            MethodDefinition[] bodies = definition.Methods.Where(method => method.Name == "MoveNext").ToArray();
            return bodies.Length == 1 ? Read(entries[0], bodies[0]) : new(YieldBodyContract.Unresolved, 0, 0);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or
            BadImageFormatException or AssemblyResolutionException or ResolutionException or InvalidOperationException)
        {
            return new(YieldBodyContract.Unresolved, 0, 0);
        }
    }

    internal static YieldBodyWindow Read(MethodDefinition entry, MethodDefinition body)
    {
        YieldBodyWindow Unknown() => new(YieldBodyContract.Unresolved, 0, 0);
        try
        {
            if (HashRuntime() != ReviewedOwnerCompletion.RuntimeHash ||
                AsyncOwnerReader.Read(entry, body) != AsyncOwnerBinding.ReturnsStateMachineTask ||
                entry.ImplAttributes != MethodImplAttributes.IL || entry.IsPInvokeImpl || entry.HasSecurityDeclarations ||
                body.ImplAttributes != MethodImplAttributes.IL || body.IsPInvokeImpl || body.HasSecurityDeclarations ||
                body.Body.ExceptionHandlers.Count != 1 || body.DeclaringType.Fields.Count != 3) return Unknown();
            var il = body.Body.Instructions;
            Code[] prefix =
            [
                Code.Ldarg_0, Code.Ldfld, Code.Stloc, Code.Ldloc, Code.Brfalse_S,
                Code.Call, Code.Stloc, Code.Ldloca, Code.Call, Code.Stloc, Code.Ldloca, Code.Call, Code.Brtrue_S,
                Code.Ldarg_0, Code.Ldc_I4_0, Code.Dup, Code.Stloc, Code.Stfld,
                Code.Ldarg_0, Code.Ldloc, Code.Stfld, Code.Ldarg_0, Code.Ldflda, Code.Ldloca, Code.Ldarg_0,
                Code.Call, Code.Leave, Code.Ldarg_0, Code.Ldfld, Code.Stloc,
                Code.Ldarg_0, Code.Ldflda, Code.Initobj, Code.Ldarg_0, Code.Ldc_I4_M1, Code.Dup,
                Code.Stloc, Code.Stfld, Code.Ldloca, Code.Call
            ];
            if (il.Count <= prefix.Length || !il.Take(prefix.Length).Select(instruction => Normalize(instruction.OpCode.Code)).SequenceEqual(prefix))
                return Unknown();
            FieldDefinition? state = Field(entry.Body.Instructions[5]);
            FieldDefinition? builder = Field(entry.Body.Instructions[2]);
            FieldDefinition? awaiter = Field(il[20]);
            if (state is null || builder is null || awaiter is null || awaiter == state || awaiter == builder || awaiter.IsStatic ||
                awaiter.DeclaringType != body.DeclaringType || !RuntimeType(awaiter.FieldType, "System.Runtime.CompilerServices.YieldAwaitable/YieldAwaiter") ||
                new[] { 1, 17, 37 }.Any(index => Field(il[index]) != state) || Field(il[22]) != builder ||
                Field(il[28]) != awaiter || Field(il[31]) != awaiter ||
                il[32].Operand is not TypeReference cleared || !RuntimeType(cleared, awaiter.FieldType.FullName)) return Unknown();
            VariableDefinition? stateLocal = Local(body, il[2]);
            VariableDefinition? awaitableLocal = Local(body, il[6]);
            VariableDefinition? awaiterLocal = Local(body, il[9]);
            if (stateLocal is null || awaitableLocal is null || awaiterLocal is null ||
                !RuntimeType(stateLocal.VariableType, "System.Int32") ||
                !RuntimeType(awaitableLocal.VariableType, "System.Runtime.CompilerServices.YieldAwaitable") ||
                !RuntimeType(awaiterLocal.VariableType, awaiter.FieldType.FullName) ||
                new[] { 3, 16, 36 }.Any(index => Local(body, il[index]) != stateLocal) || Local(body, il[7]) != awaitableLocal ||
                new[] { 10, 19, 23, 29, 38 }.Any(index => Local(body, il[index]) != awaiterLocal) ||
                il[4].Operand != il[27] || il[12].Operand != il[38] || il[26].Operand != il[^1] ||
                !Call(il[5], "System.Runtime.CompilerServices.YieldAwaitable System.Threading.Tasks.Task::Yield()") ||
                !Call(il[8], "System.Runtime.CompilerServices.YieldAwaitable/YieldAwaiter System.Runtime.CompilerServices.YieldAwaitable::GetAwaiter()") ||
                !Call(il[11], "System.Boolean System.Runtime.CompilerServices.YieldAwaitable/YieldAwaiter::get_IsCompleted()") ||
                !Call(il[39], "System.Void System.Runtime.CompilerServices.YieldAwaitable/YieldAwaiter::GetResult()") ||
                !Await(il[25], body.DeclaringType)) return Unknown();
            ExceptionHandler handler = body.Body.ExceptionHandlers[0];
            int handlerStart = il.IndexOf(handler.HandlerStart);
            int completion = il.IndexOf(handler.HandlerEnd);
            if (handler.HandlerType != ExceptionHandlerType.Catch || handler.FilterStart is not null ||
                handler.CatchType is null || !RuntimeType(handler.CatchType, "System.Exception") || handler.TryStart != il[3] ||
                handler.TryEnd != handler.HandlerStart || handlerStart <= prefix.Length || completion != handlerStart + 9 ||
                il.Count != completion + 7 || Normalize(il[handlerStart - 1].OpCode.Code) != Code.Leave ||
                il[handlerStart - 1].Operand != il[completion]) return Unknown();
            Code[] catchCodes = [Code.Stloc, Code.Ldarg_0, Code.Ldc_I4_S, Code.Stfld, Code.Ldarg_0, Code.Ldflda, Code.Ldloc, Code.Call, Code.Leave];
            Code[] completionCodes = [Code.Ldarg_0, Code.Ldc_I4_S, Code.Stfld, Code.Ldarg_0, Code.Ldflda, Code.Call, Code.Ret];
            VariableDefinition? exceptionLocal = Local(body, il[handlerStart]);
            if (!il.Skip(handlerStart).Take(9).Select(instruction => Normalize(instruction.OpCode.Code)).SequenceEqual(catchCodes) ||
                !il.Skip(completion).Select(instruction => Normalize(instruction.OpCode.Code)).SequenceEqual(completionCodes) ||
                exceptionLocal is null || !RuntimeType(exceptionLocal.VariableType, "System.Exception") ||
                Local(body, il[handlerStart + 6]) != exceptionLocal ||
                il[handlerStart + 2].Operand is not sbyte failureState || failureState != -2 ||
                il[completion + 1].Operand is not sbyte completedState || completedState != -2 ||
                Field(il[handlerStart + 3]) != state || Field(il[completion + 2]) != state ||
                Field(il[handlerStart + 5]) != builder || Field(il[completion + 4]) != builder ||
                il[handlerStart + 8].Operand != il[^1] ||
                !Call(il[handlerStart + 7], "System.Void System.Runtime.CompilerServices.AsyncTaskMethodBuilder::SetException(System.Exception)") ||
                !Call(il[completion + 5], "System.Void System.Runtime.CompilerServices.AsyncTaskMethodBuilder::SetResult()")) return Unknown();
            // Body control flow must not re-enter the scheduling frame, escape
            // the observed completion path, or overwrite its frame state.
            for (int index = prefix.Length; index < handlerStart - 1; index++)
            {
                Instruction instruction = il[index];
                if (instruction.Operand is Instruction target && !Inside(target) ||
                    instruction.Operand is Instruction[] targets && targets.Any(target => !Inside(target)) ||
                    instruction.Operand is FieldReference field && field.Resolve() is FieldDefinition actual &&
                        (actual == state || actual == builder || actual == awaiter) ||
                    Local(body, instruction) is VariableDefinition local &&
                        (local == stateLocal || local == awaitableLocal || local == awaiterLocal || local == exceptionLocal) ||
                    instruction.OpCode.Code is Code.Ret or Code.Leave or Code.Leave_S or Code.Jmp or Code.Endfinally or Code.Endfilter or
                        Code.Ldarg_0 or Code.Ldarg_1 or Code.Ldarg_2 or Code.Ldarg_3 or Code.Ldarg or Code.Ldarg_S or Code.Ldarga or Code.Ldarga_S or Code.Starg or Code.Starg_S)
                    return Unknown();
            }
            bool Inside(Instruction instruction) => il.IndexOf(instruction) is int index && index >= prefix.Length && index < handlerStart;
            return HashRuntime() == ReviewedOwnerCompletion.RuntimeHash
                ? new(YieldBodyContract.SingleYieldTaskBody, prefix.Length, handlerStart - prefix.Length - 1) : Unknown();
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or
            BadImageFormatException or AssemblyResolutionException or ResolutionException or InvalidOperationException)
        {
            return Unknown();
        }
    }

    private static bool Await(Instruction instruction, TypeDefinition state)
    {
        if (instruction.Operand is not GenericInstanceMethod call || call.GenericArguments.Count != 2 ||
            !RuntimeType(call.GenericArguments[0], "System.Runtime.CompilerServices.YieldAwaitable/YieldAwaiter") ||
            call.GenericArguments[1].Resolve() != state || call.ElementMethod.Resolve() is not MethodDefinition definition ||
            !Runtime(definition) || definition.Name != "AwaitUnsafeOnCompleted" ||
            definition.DeclaringType.FullName != "System.Runtime.CompilerServices.AsyncTaskMethodBuilder" ||
            !RuntimeType(call.DeclaringType, definition.DeclaringType.FullName) || call.Name != definition.Name || definition.IsStatic ||
            definition.GenericParameters.Count != 2 || definition.Parameters.Count != 2 ||
            !call.HasThis || call.ExplicitThis || call.ElementMethod.GenericParameters.Count != 2 ||
            call.Parameters.Count != 2 || !RuntimeType(call.ReturnType, "System.Void") ||
            call.CallingConvention != MethodCallingConvention.Generic) return false;
        return definition.Parameters.Select((parameter, index) => parameter.ParameterType is ByReferenceType byReference &&
            byReference.ElementType is GenericParameter argument && argument.Type == GenericParameterType.Method &&
            argument.Position == index && ReferenceEquals(argument.Owner, definition)).All(value => value) &&
            call.Parameters.Select((parameter, index) => parameter.ParameterType is ByReferenceType byReference &&
            byReference.ElementType is GenericParameter argument && argument.Type == GenericParameterType.Method &&
            argument.Position == index && ReferenceEquals(argument.Owner, call.ElementMethod)).All(value => value);
    }

    private static bool Call(Instruction instruction, string signature) => instruction.Operand is MethodReference call &&
        call is not GenericInstanceMethod && !call.HasGenericParameters && !call.ExplicitThis && call.CallingConvention == MethodCallingConvention.Default &&
        call.FullName == signature && call.Resolve() is MethodDefinition definition && Runtime(definition) &&
        definition.FullName == signature && definition.HasThis == call.HasThis && RuntimeType(call.ReturnType, definition.ReturnType.FullName) &&
        call.Parameters.All(parameter => RuntimeType(parameter.ParameterType, parameter.ParameterType.FullName));

    private static bool RuntimeType(TypeReference type, string name) => type is not TypeSpecification && type.FullName == name &&
        type.Resolve() is TypeDefinition definition && definition.FullName == name && SameRuntime(definition.Module.FileName);
    private static bool Runtime(MethodDefinition method) => SameRuntime(method.Module.FileName);
    private static bool SameRuntime(string file) => string.Equals(Path.GetFullPath(file), Path.GetFullPath(typeof(object).Assembly.Location),
        OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
    private static string HashRuntime()
    {
        using var stream = File.OpenRead(typeof(object).Assembly.Location);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }
    private static FieldDefinition? Field(Instruction instruction) => instruction.Operand is FieldReference field ? field.Resolve() : null;
    private static VariableDefinition? Local(MethodDefinition method, Instruction instruction)
    {
        if (instruction.Operand is VariableDefinition variable) return variable;
        int index = instruction.OpCode.Code switch
        {
            Code.Ldloc_0 or Code.Stloc_0 => 0, Code.Ldloc_1 or Code.Stloc_1 => 1,
            Code.Ldloc_2 or Code.Stloc_2 => 2, Code.Ldloc_3 or Code.Stloc_3 => 3, _ => -1
        };
        return index >= 0 && index < method.Body.Variables.Count ? method.Body.Variables[index] : null;
    }
    private static Code Normalize(Code code) => code switch
    {
        Code.Ldloc_0 or Code.Ldloc_1 or Code.Ldloc_2 or Code.Ldloc_3 or Code.Ldloc_S => Code.Ldloc,
        Code.Stloc_0 or Code.Stloc_1 or Code.Stloc_2 or Code.Stloc_3 or Code.Stloc_S => Code.Stloc,
        Code.Ldloca_S => Code.Ldloca, Code.Brtrue => Code.Brtrue_S, Code.Brfalse => Code.Brfalse_S,
        Code.Leave_S => Code.Leave, _ => code
    };
}
