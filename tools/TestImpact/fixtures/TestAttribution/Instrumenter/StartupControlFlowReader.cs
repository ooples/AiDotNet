using AiDotNet.TestImpact;
using Mono.Cecil;
using Mono.Cecil.Cil;

internal enum StartupOperation { ObserveEntry, Parallelism, ObserveResetInput, Reset, ObserveResetCompletion, OverrideKey, Sign, ObserveCompletion }
internal sealed record StartupCall(StartupOperation Operation, MethodReference Method);
internal sealed record StartupControlFlow(StartupCall[] Calls, FieldDefinition Initialized, FieldReference Key,
    MethodReference[] ReachedMethods, TypeReference[] ReachedTypes);

// Evaluates only the observed default-license/default-MDOP branch. Recognition
// is NOT authentication: the caller must bind every recorded target, the helper
// initializer, the module entry point, and host lifetime before using the result.
internal static class StartupControlFlowReader
{
    private enum ValueKind { Null, Integer, Text, Cpu, Type, ResetInput, Key, Signature }
    private sealed record Value(ValueKind Kind, int Integer = 0, string Text = "", TypeReference? Type = null, FieldReference? Field = null);

    internal static StartupControlFlow? ReadShape(MethodDefinition method, RuntimeContractProfile? profile)
    {
        try
        {
            if (!Inputs(profile) || !method.IsStatic || method.HasThis || method.ExplicitThis || method.HasGenericParameters ||
                method.Parameters.Count != 0 || method.CallingConvention != MethodCallingConvention.Default || !method.HasBody ||
                method.ImplAttributes != MethodImplAttributes.IL || method.IsPInvokeImpl || method.HasSecurityDeclarations ||
                !SignedLicenseReader.Type(method.ReturnType, typeof(void)) || method.DeclaringType.HasGenericParameters ||
                method.DeclaringType.Methods.Any(candidate => candidate.IsConstructor) || method.DeclaringType.Fields.Count != 1) return null;
            FieldDefinition initialized = method.DeclaringType.Fields[0];
            if (!initialized.IsPrivate || !initialized.IsStatic || initialized.IsInitOnly || initialized.IsLiteral || initialized.HasCustomAttributes ||
                initialized.InitialValue.Length != 0 || !SignedLicenseReader.Type(initialized.FieldType, typeof(bool))) return null;
            var il = method.Body.Instructions;
            var stack = new Stack<Value>();
            var locals = new Dictionary<int, Value>();
            var calls = new List<StartupCall>();
            var writes = new HashSet<string>(StringComparer.Ordinal);
            var visited = new HashSet<int>();
            FieldReference? key = null;
            bool flag = false;
            int pc = 0;
            Value Pop() => stack.Count != 0 ? stack.Pop() : throw new InvalidDataException("Startup stack underflow.");
            int Number(Value value) => value.Kind == ValueKind.Integer ? value.Integer : throw new InvalidDataException("Expected startup integer.");
            string Text(Value value) => value.Kind == ValueKind.Text ? value.Text : throw new InvalidDataException("Expected startup text.");
            bool Truth(Value value) => value.Kind switch { ValueKind.Integer => value.Integer != 0, ValueKind.Null => false,
                ValueKind.Cpu or ValueKind.Type or ValueKind.ResetInput or ValueKind.Key or ValueKind.Signature or ValueKind.Text => true,
                _ => throw new InvalidDataException("Unknown startup branch value.") };
            int Target(Instruction instruction) => instruction.Operand is Instruction target && il.IndexOf(target) >= 0
                ? il.IndexOf(target) : throw new InvalidDataException("Foreign startup branch.");
            int Local(Instruction instruction) => instruction.Operand is VariableDefinition variable && method.Body.Variables.Contains(variable)
                ? variable.Index : throw new InvalidDataException("Foreign startup local.");
            void Store(int index)
            {
                if (index < 0 || index >= method.Body.Variables.Count) throw new InvalidDataException("Missing startup local.");
                Value value = Pop();
                TypeReference type = method.Body.Variables[index].VariableType;
                bool valid = value.Kind switch
                {
                    ValueKind.Integer => SignedLicenseReader.Type(type, typeof(int)) ||
                        value.Integer is 0 or 1 && SignedLicenseReader.Type(type, typeof(bool)),
                    ValueKind.Text => SignedLicenseReader.Type(type, typeof(string)),
                    ValueKind.Cpu => type is not TypeSpecification && type.FullName == "AiDotNet.Tensors.Engines.IEngine",
                    _ => false
                };
                if (!valid) throw new InvalidDataException("Mismatched startup local.");
                locals[index] = value;
            }
            while (pc < il.Count && visited.Add(pc))
            {
                Instruction instruction = il[pc++];
                switch (instruction.OpCode.Code)
                {
                    case Code.Ldc_I4_0: stack.Push(new(ValueKind.Integer, 0)); break;
                    case Code.Ldc_I4_1: stack.Push(new(ValueKind.Integer, 1)); break;
                    case Code.Ldc_I4_2: stack.Push(new(ValueKind.Integer, 2)); break;
                    case Code.Ldc_I4_3: stack.Push(new(ValueKind.Integer, 3)); break;
                    case Code.Ldnull: stack.Push(new(ValueKind.Null)); break;
                    case Code.Ldstr: stack.Push(new(ValueKind.Text, Text: (string)instruction.Operand)); break;
                    case Code.Dup: stack.Push(stack.Peek()); break;
                    case Code.Pop: Pop(); break;
                    case Code.Stloc_0: Store(0); break;
                    case Code.Stloc_1: Store(1); break;
                    case Code.Stloc_2: Store(2); break;
                    case Code.Stloc_3: Store(3); break;
                    case Code.Stloc_S: case Code.Stloc: Store(Local(instruction)); break;
                    case Code.Ldloc_0: stack.Push(locals[0]); break;
                    case Code.Ldloc_1: stack.Push(locals[1]); break;
                    case Code.Ldloc_2: stack.Push(locals[2]); break;
                    case Code.Ldloc_3: stack.Push(locals[3]); break;
                    case Code.Ldloc_S: case Code.Ldloc: stack.Push(locals[Local(instruction)]); break;
                    case Code.Br_S: case Code.Br: pc = Target(instruction); break;
                    case Code.Brtrue_S: case Code.Brtrue: if (Truth(Pop())) pc = Target(instruction); break;
                    case Code.Brfalse_S: case Code.Brfalse: if (!Truth(Pop())) pc = Target(instruction); break;
                    case Code.Leave_S: case Code.Leave: if (stack.Count != 0) return null; pc = Target(instruction); break;
                    case Code.Ceq: { Value right = Pop(); stack.Push(new(ValueKind.Integer, Pop() == right ? 1 : 0)); break; }
                    case Code.Ldtoken:
                        if (instruction.Operand is not TypeReference type || type.FullName != "AiDotNet.Tensors.Engines.CpuEngine") return null;
                        stack.Push(new(ValueKind.Type, Type: type)); break;
                    case Code.Isinst:
                        if (instruction.Operand is not TypeReference testType || testType.FullName != "AiDotNet.Tensors.Engines.CpuEngine" || Pop().Kind != ValueKind.Cpu) return null;
                        stack.Push(new(ValueKind.Cpu, Type: testType)); break;
                    case Code.Ldsfld:
                        if (OwnedFieldBinding.Read(instruction, method.DeclaringType) == initialized) stack.Push(new(ValueKind.Integer, flag ? 1 : 0));
                        else if (instruction.Operand is FieldReference field && field.Resolve() is FieldDefinition definition && definition.IsStatic && definition.IsInitOnly &&
                            SignedLicenseReader.Type(field.FieldType, typeof(byte[])))
                        { if (key is not null && key.Resolve() != definition) return null; key = field; stack.Push(new(ValueKind.Key, Field: field)); }
                        else return null;
                        break;
                    case Code.Stsfld:
                        if (OwnedFieldBinding.Read(instruction, method.DeclaringType) != initialized || flag || Number(Pop()) != 1) return null;
                        flag = true; break;
                    case Code.Call: case Code.Callvirt: case Code.Newobj:
                        if (instruction.Operand is not MethodReference call || !Invoke(call, instruction.OpCode.Code)) return null;
                        break;
                    case Code.Ret:
                        if (stack.Count != 0 || !flag || key is null ||
                            !calls.Select(call => call.Operation).SequenceEqual(Enum.GetValues<StartupOperation>()) ||
                            !writes.SetEquals(["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "AIDOTNET_LICENSE_KEY"])) return null;
                        // The only catch is the CPU reset's empty catch. The
                        // observed post-call marker excludes this exceptional
                        // edge; no filter/finally or alternate catch is ignored.
                        if (method.Body.ExceptionHandlers.Count != 1) return null;
                        var handler = method.Body.ExceptionHandlers[0];
                        int start = il.IndexOf(handler.HandlerStart), end = il.IndexOf(handler.HandlerEnd);
                        if (handler.HandlerType != ExceptionHandlerType.Catch || handler.FilterStart is not null || handler.CatchType is null ||
                            !SignedLicenseReader.Type(handler.CatchType, typeof(object)) || start < 0 || end != start + 2 ||
                            il[start].OpCode.Code != Code.Pop || il[start + 1].OpCode.Code is not (Code.Leave or Code.Leave_S) ||
                            handler.TryEnd != handler.HandlerStart || Target(il[start + 1]) != end ||
                            !visited.Contains(il.IndexOf(handler.TryStart)) || visited.Contains(start)) return null;
                        int resetIndex = visited.Single(index => il[index].Operand == calls[(int)StartupOperation.Reset].Method);
                        int completionIndex = visited.Single(index => il[index].Operand == calls[(int)StartupOperation.ObserveResetCompletion].Method);
                        if (resetIndex < il.IndexOf(handler.TryStart) || completionIndex <= resetIndex || completionIndex >= start) return null;
                        return new(calls.ToArray(), initialized, key,
                            visited.Select(index => il[index].Operand).OfType<MethodReference>().ToArray(),
                            visited.Select(index => il[index].Operand).OfType<TypeReference>().ToArray());
                    default: return null;
                }
            }
            return null;

            bool Invoke(MethodReference call, Code opcode)
            {
                if (call is MethodSpecification || call.HasGenericParameters || call.ExplicitThis || call.CallingConvention != MethodCallingConvention.Default) return false;
                string name = call.FullName;
                StartupOperation? operation = name switch
                {
                    "System.Void AiDotNet.TestImpact.Xunit.RuntimeContractInitialization::RecordCpuStartup()" => StartupOperation.ObserveEntry,
                    "System.Void AiDotNet.Tensors.Helpers.CpuParallelSettings::set_MaxDegreeOfParallelism(System.Int32)" => StartupOperation.Parallelism,
                    "System.Void AiDotNet.TestImpact.Xunit.RuntimeContractInitialization::RecordCpuResetInput(AiDotNet.TestImpact.RuntimeCpuResetInput)" => StartupOperation.ObserveResetInput,
                    "System.Void AiDotNet.Tensors.Engines.AiDotNetEngine::ResetToCpu()" => StartupOperation.Reset,
                    "System.Void AiDotNet.TestImpact.Xunit.RuntimeContractInitialization::RecordCpuResetCompletion()" => StartupOperation.ObserveResetCompletion,
                    "System.Void AiDotNet.Helpers.BuildKeyProvider::OverrideForTesting(System.Byte[])" => StartupOperation.OverrideKey,
                    "System.String AiDotNet.Tests.Helpers.LicenseTestSupport::SignedKey(System.String,System.Byte[])" => StartupOperation.Sign,
                    "System.Void AiDotNet.TestImpact.Xunit.RuntimeContractInitialization::RecordCpuCompletion(System.Boolean,System.Int32)" => StartupOperation.ObserveCompletion,
                    _ => null
                };
                if (operation is StartupOperation step)
                {
                    if (opcode != Code.Call || call.HasThis || calls.Count != (int)step) return false;
                    switch (step)
                    {
                        case StartupOperation.ObserveEntry: if (flag || writes.Count != 0) return false; break;
                        case StartupOperation.Parallelism: if (Number(Pop()) != 1) return false; break;
                        case StartupOperation.ObserveResetInput: if (Pop().Kind != ValueKind.ResetInput) return false; break;
                        case StartupOperation.OverrideKey: if (Pop().Kind != ValueKind.Key) return false; break;
                        case StartupOperation.Sign:
                            if (Pop().Kind != ValueKind.Null || Text(Pop()) != "testdefault1") return false;
                            stack.Push(new(ValueKind.Signature)); break;
                        case StartupOperation.ObserveCompletion: if (Number(Pop()) != 1 || Number(Pop()) != 1) return false; break;
                    }
                    calls.Add(new(step, call)); return true;
                }
                switch (name)
                {
                    case "System.String System.Environment::GetEnvironmentVariable(System.String)":
                        string variable = Text(Pop());
                        if (writes.Contains(variable) || variable is not ("AIDOTNET_TEST_CPU_MDOP" or "AIDOTNET_LICENSE_KEY" or "AIDOTNET_QUIET")) return false;
                        stack.Push(new(ValueKind.Text, Text: variable == "AIDOTNET_QUIET" ? "1" : "")); return opcode == Code.Call && !call.HasThis;
                    case "System.Void System.Environment::SetEnvironmentVariable(System.String,System.String)":
                        Value value = Pop(); string keyName = Text(Pop());
                        if (keyName == "AIDOTNET_LICENSE_KEY" ? value.Kind != ValueKind.Signature :
                            keyName is not ("OMP_NUM_THREADS" or "MKL_NUM_THREADS" or "OPENBLAS_NUM_THREADS") || Text(value) != "1") return false;
                        return writes.Add(keyName) && opcode == Code.Call && !call.HasThis;
                    case "System.Boolean System.String::IsNullOrWhiteSpace(System.String)":
                    case "System.Boolean System.String::IsNullOrEmpty(System.String)":
                        Value text = Pop(); if (text.Kind is not (ValueKind.Text or ValueKind.Null)) return false;
                        stack.Push(new(ValueKind.Integer, text.Kind == ValueKind.Null || string.IsNullOrEmpty(text.Text) ? 1 : 0));
                        return opcode == Code.Call && !call.HasThis;
                    case "AiDotNet.Tensors.Engines.IEngine AiDotNet.Tensors.Engines.AiDotNetEngine::get_Current()":
                        stack.Push(new(ValueKind.Cpu)); return opcode == Code.Call && !call.HasThis;
                    case "System.Type System.Object::GetType()":
                        if (Pop().Kind != ValueKind.Cpu) return false;
                        stack.Push(new(ValueKind.Type)); return opcode is Code.Call or Code.Callvirt && call.HasThis;
                    case "System.Type System.Type::GetTypeFromHandle(System.RuntimeTypeHandle)":
                        Value handle = Pop(); if (handle.Kind != ValueKind.Type || handle.Type is null) return false;
                        stack.Push(handle); return opcode == Code.Call && !call.HasThis;
                    case "System.Boolean System.Type::op_Equality(System.Type,System.Type)":
                        if (Pop().Kind != ValueKind.Type || Pop().Kind != ValueKind.Type) return false;
                        stack.Push(new(ValueKind.Integer, 1)); return opcode == Code.Call && !call.HasThis;
                    case "System.Void AiDotNet.TestImpact.RuntimeCpuResetInput::.ctor(AiDotNet.TestImpact.RuntimeCpuEntryMode,AiDotNet.TestImpact.RuntimeCpuLogging)":
                        if (Number(Pop()) != 1 || Number(Pop()) != 1) return false;
                        stack.Push(new(ValueKind.ResetInput)); return opcode == Code.Newobj && call.HasThis;
                    case "System.Int32 AiDotNet.Tensors.Helpers.CpuParallelSettings::get_MaxDegreeOfParallelism()":
                        if (!calls.Any(item => item.Operation == StartupOperation.Parallelism)) return false;
                        stack.Push(new(ValueKind.Integer, 1)); return opcode == Code.Call && !call.HasThis;
                    default: return false;
                }
            }
        }
        catch (Exception error) when (error is IOException or InvalidDataException or UnauthorizedAccessException or ArgumentException or BadImageFormatException or
            AssemblyResolutionException or ResolutionException or InvalidOperationException or KeyNotFoundException or InvalidCastException)
        { return null; }
    }

    internal static bool Inputs(RuntimeContractProfile? profile) => RuntimeProfileEvidence.HasObservedSuccessfulCpuReset(profile) && profile?.Initialization is
    {
        Inputs: { LicenseStartup: RuntimeLicenseStartupPolicy.DefaultTestLicense, CpuParallelism: RuntimeCpuParallelismPolicy.DefaultSingleThread },
        Completion.MaxDegreeOfParallelism: 1
    };
}
