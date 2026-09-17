using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;

internal enum ConstructorKeyFact { Unknown, NonWhitespaceString }
internal enum ConstructorPrefixContract { Unresolved, ReceiverFieldsAndBaseConstructor }
internal sealed record ConstructorPrefixAssessment(ConstructorPrefixContract Contract, string BaseConstructor, string[] WrittenFields);

// Evaluates one concrete normal input path before the separately reviewed lock
// tail. It records, rather than discharges, the base constructor. In particular
// a known nonempty key must come from a caller proof, not a parameter's name.
internal static class ConstructorPrefixReader
{
    private enum Kind { Receiver, Integer, Key }
    private readonly record struct Value(Kind Kind, int Integer = 0);

    internal static ConstructorPrefixAssessment Read(MethodDefinition method, int rank, int worldSize, ConstructorKeyFact keyFact)
    {
        ConstructorPrefixAssessment Unknown() => new(ConstructorPrefixContract.Unresolved, "", []);
        try
        {
            LockedInitializationAssessment locked = LockedInitializationReader.Read(method);
            if (locked.Contract != LockedInitializationContract.ConstantInsertIfAbsent || keyFact != ConstructorKeyFact.NonWhitespaceString ||
                method.Parameters.Count != 3 || !Runtime(method.Parameters[0].ParameterType, "System.Int32") ||
                !Runtime(method.Parameters[1].ParameterType, "System.Int32") || !Runtime(method.Parameters[2].ParameterType, "System.String") ||
                method.DeclaringType.BaseType is null) return Unknown();
            var stack = new Stack<Value>();
            var fields = new Dictionary<string, Value>(StringComparer.Ordinal);
            var il = method.Body.Instructions;
            string? baseConstructor = null;
            int index = 0, steps = 0;
            while (index < locked.Start)
            {
                if (++steps > 256) return Unknown();
                Instruction instruction = il[index];
                int next = index + 1;
                switch (instruction.OpCode.Code)
                {
                    case Code.Ldarg_0: stack.Push(new(Kind.Receiver)); break;
                    case Code.Ldarg_1: stack.Push(new(Kind.Integer, rank)); break;
                    case Code.Ldarg_2: stack.Push(new(Kind.Integer, worldSize)); break;
                    case Code.Ldarg_3: stack.Push(new(Kind.Key)); break;
                    case Code.Ldc_I4_0: stack.Push(new(Kind.Integer, 0)); break;
                    case Code.Ldc_I4_1: stack.Push(new(Kind.Integer, 1)); break;
                    case Code.Ldc_I4_M1: stack.Push(new(Kind.Integer, -1)); break;
                    case Code.Ldc_I4: stack.Push(new(Kind.Integer, (int)instruction.Operand)); break;
                    case Code.Ldc_I4_S: stack.Push(new(Kind.Integer, (sbyte)instruction.Operand)); break;
                    case Code.Call:
                    {
                        if (instruction.Operand is not MethodReference call || call.ExplicitThis || call.HasGenericParameters ||
                            call.CallingConvention != MethodCallingConvention.Default || call.Resolve() is not MethodDefinition definition ||
                            !stack.TryPop(out Value receiver)) return Unknown();
                        if (call.HasThis && receiver.Kind == Kind.Receiver && definition.IsConstructor && !definition.IsStatic &&
                            call.Name == ".ctor" && call.Parameters.Count == 0 && definition.Parameters.Count == 0 &&
                            call.DeclaringType.FullName == method.DeclaringType.BaseType.FullName &&
                            call.DeclaringType.Resolve() == method.DeclaringType.BaseType.Resolve() && Runtime(call.ReturnType, "System.Void") &&
                            baseConstructor is null && fields.Count == 0 && stack.Count == 0)
                        {
                            baseConstructor = call.FullName;
                            break;
                        }
                        if (call.HasThis || receiver.Kind != Kind.Key || call.FullName != "System.Boolean System.String::IsNullOrWhiteSpace(System.String)" ||
                            !Runtime(call.DeclaringType, "System.String") || !Runtime(call.ReturnType, "System.Boolean") ||
                            call.Parameters.Count != 1 || !Runtime(call.Parameters[0].ParameterType, "System.String") ||
                            definition.FullName != call.FullName || definition.HasThis) return Unknown();
                        stack.Push(new(Kind.Integer, 0));
                        break;
                    }
                    case Code.Stfld:
                    {
                        if (baseConstructor is null || OwnedFieldBinding.Read(instruction, method.DeclaringType) is not FieldDefinition field ||
                            field.DeclaringType != method.DeclaringType || field.IsStatic || !field.IsPrivate || !field.IsInitOnly || field.HasCustomAttributes ||
                            !stack.TryPop(out Value value) || !stack.TryPop(out Value receiver) || receiver.Kind != Kind.Receiver ||
                            !(value.Kind == Kind.Integer && Runtime(field.FieldType, "System.Int32") || value.Kind == Kind.Key && Runtime(field.FieldType, "System.String")) ||
                            !fields.TryAdd(field.FullName, value)) return Unknown();
                        break;
                    }
                    case Code.Blt: case Code.Blt_S: case Code.Bgt: case Code.Bgt_S:
                    case Code.Bge: case Code.Bge_S: case Code.Ble: case Code.Ble_S:
                    case Code.Beq: case Code.Beq_S: case Code.Bne_Un: case Code.Bne_Un_S:
                    {
                        if (!stack.TryPop(out Value right) || !stack.TryPop(out Value left) || left.Kind != Kind.Integer || right.Kind != Kind.Integer ||
                            instruction.Operand is not Instruction target) return Unknown();
                        bool taken = instruction.OpCode.Code switch
                        {
                            Code.Blt or Code.Blt_S => left.Integer < right.Integer,
                            Code.Bgt or Code.Bgt_S => left.Integer > right.Integer,
                            Code.Bge or Code.Bge_S => left.Integer >= right.Integer,
                            Code.Ble or Code.Ble_S => left.Integer <= right.Integer,
                            Code.Beq or Code.Beq_S => left.Integer == right.Integer,
                            _ => left.Integer != right.Integer
                        };
                        if (taken) next = il.IndexOf(target);
                        break;
                    }
                    case Code.Brtrue: case Code.Brtrue_S: case Code.Brfalse: case Code.Brfalse_S:
                    {
                        if (!stack.TryPop(out Value condition) || condition.Kind != Kind.Integer || instruction.Operand is not Instruction target) return Unknown();
                        bool truthy = instruction.OpCode.Code is Code.Brtrue or Code.Brtrue_S;
                        if ((condition.Integer != 0) == truthy) next = il.IndexOf(target);
                        break;
                    }
                    case Code.Br: case Code.Br_S:
                        if (instruction.Operand is not Instruction jump) return Unknown();
                        next = il.IndexOf(jump); break;
                    default: return Unknown();
                }
                if (next <= index || next > locked.Start) return Unknown();
                index = next;
            }
            if (stack.Count != 0 || baseConstructor is null || !fields.TryGetValue(locked.Key, out Value writtenKey) || writtenKey.Kind != Kind.Key)
                return Unknown();
            using var stream = File.OpenRead(typeof(object).Assembly.Location);
            return Convert.ToHexStringLower(SHA256.HashData(stream)) == ReviewedOwnerCompletion.RuntimeHash
                ? new(ConstructorPrefixContract.ReceiverFieldsAndBaseConstructor, baseConstructor, fields.Keys.Order(StringComparer.Ordinal).ToArray()) : Unknown();
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or
            BadImageFormatException or AssemblyResolutionException or ResolutionException or InvalidOperationException or InvalidCastException)
        {
            return Unknown();
        }
    }

    private static bool Runtime(TypeReference reference, string name) => reference is not TypeSpecification && reference.FullName == name &&
        reference.Resolve() is TypeDefinition definition && definition.FullName == name &&
        string.Equals(Path.GetFullPath(definition.Module.FileName), Path.GetFullPath(typeof(object).Assembly.Location),
            OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
}
