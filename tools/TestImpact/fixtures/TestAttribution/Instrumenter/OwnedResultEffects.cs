using System.Security.Cryptography;
using System.Text.Json;
using AiDotNet.TestImpact;
using Mono.Cecil;
using Mono.Cecil.Cil;

// Small abstract interpreter for the first runtime-effects experiment. It proves
// that eligible boolean writes affect a fresh returned object which did not
// escape through its constructor/setters. It does not close opaque callbacks or
// claim that unchanged constructor/runtime effects are globally pure.
internal sealed class OwnedResultEffects
{
    [Flags]
    private enum Origin { Other = 1, Receiver = 2, Fresh = 4 }
    private readonly record struct Value(Origin Origin, int Producer = -1);
    private sealed record State(List<Value> Stack, Value[] Locals)
    {
        internal bool AfterOwnedWrite { get; set; }
        internal State Copy() => new(new(Stack), (Value[])Locals.Clone()) { AfterOwnedWrite = AfterOwnedWrite };
    }
    private readonly Dictionary<string, string[]?> constructors = new(StringComparer.Ordinal);

    internal OwnedResultEffect? Read(MethodDefinition method)
    {
        if (!method.IsStatic || !SupportedBody(method) || method.ReturnType is GenericParameter or ArrayType ||
            method.Body.Instructions.Count(instruction => instruction.OpCode.Code == Code.Newobj &&
                instruction.Operand is MethodReference constructor && constructor.DeclaringType.FullName == method.ReturnType.FullName) != 1) return null;
        TypeDefinition? owner = Resolve(method.ReturnType);
        if (owner is null || owner.IsValueType || HasFinalizer(owner)) return null;
        var dependencies = new HashSet<string>(StringComparer.Ordinal);
        var writes = new Dictionary<int, OwnedBooleanWrite>();
        if (!Analyze(method, owner, false, dependencies, writes, new(StringComparer.Ordinal)) || writes.Count == 0) return null;
        var shape = new
        {
            method.Body.InitLocals,
            Locals = method.Body.Variables.Select(variable => variable.VariableType.FullName + "@" + variable.VariableType.Scope).ToArray(),
            Instructions = method.Body.Instructions.Select((instruction, index) => new
            {
                Op = writes.ContainsKey(index) ? Code.Ldc_I4 : instruction.OpCode.Code,
                OwnedBoolean = writes.ContainsKey(index),
                Operand = writes.ContainsKey(index) ? null : Operand(instruction.Operand, method)
            }).ToArray()
        };
        return new(1, RuntimeEffectScope.FreshReturnedObject, Convert.ToHexStringLower(SHA256.HashData(JsonSerializer.SerializeToUtf8Bytes(shape))),
            dependencies.Order(StringComparer.Ordinal).ToArray(), writes.Values.OrderBy(write => write.Instruction).ToArray());
    }

    private bool PrivateConstructor(MethodDefinition method, TypeDefinition owner, HashSet<string> dependencies, HashSet<string> active)
    {
        string key = DependencyGraph.Stable(method) + "@" + owner.Module.Assembly.Name.FullName + ":" + owner.FullName;
        if (constructors.TryGetValue(key, out string[]? cached))
        {
            if (cached is null) return false;
            dependencies.UnionWith(cached);
            return true;
        }
        if (!SupportedBody(method) || active.Count >= 8 || !active.Add(key)) return false;
        var found = new HashSet<string>(StringComparer.Ordinal) { DependencyGraph.Stable(method) };
        bool valid = Analyze(method, owner, true, found, new(), active);
        active.Remove(key);
        constructors[key] = valid ? found.ToArray() : null;
        if (valid) dependencies.UnionWith(found);
        return valid;
    }

    private bool Analyze(MethodDefinition method, TypeDefinition owner, bool constructor,
        HashSet<string> dependencies, Dictionary<int, OwnedBooleanWrite> writes, HashSet<string> active)
    {
        var instructions = method.Body.Instructions;
        var states = new Dictionary<int, State>();
        var pending = new Queue<int>();
        states.Add(0, new([], Enumerable.Repeat(new Value(Origin.Other), method.Body.Variables.Count).ToArray()));
        pending.Enqueue(0);
        bool returned = false;
        int iterations = 0;
        while (pending.TryDequeue(out int index))
        {
            if (++iterations > 1024) return false;
            State state = states[index].Copy();
            Instruction instruction = instructions[index];
            bool terminal = false;
            var branches = new List<int>();
            bool Pop(out Value value)
            {
                if (state.Stack.Count == 0) { value = default; return false; }
                value = state.Stack[^1]; state.Stack.RemoveAt(state.Stack.Count - 1); return true;
            }
            void Push(Value value) => state.Stack.Add(value);
            switch (instruction.OpCode.Code)
            {
                case Code.Nop: break;
                case Code.Ldarg_0: case Code.Ldarg_1: case Code.Ldarg_2: case Code.Ldarg_3:
                case Code.Ldarg: case Code.Ldarg_S:
                {
                    int argument = instruction.Operand is ParameterDefinition parameter ? parameter.Index + (method.HasThis ? 1 : 0)
                        : (int)instruction.OpCode.Code - (int)Code.Ldarg_0;
                    Push(new(constructor && argument == 0 ? Origin.Receiver : Origin.Other, index)); break;
                }
                case Code.Ldnull: case Code.Ldstr:
                case Code.Ldc_I4_M1: case Code.Ldc_I4_0: case Code.Ldc_I4_1: case Code.Ldc_I4_2: case Code.Ldc_I4_3:
                case Code.Ldc_I4_4: case Code.Ldc_I4_5: case Code.Ldc_I4_6: case Code.Ldc_I4_7: case Code.Ldc_I4_8:
                case Code.Ldc_I4: case Code.Ldc_I4_S: case Code.Ldc_I8: case Code.Ldc_R4: case Code.Ldc_R8:
                    Push(new(Origin.Other, index)); break;
                case Code.Dup:
                    if (!Pop(out Value duplicated)) return false; Push(duplicated); Push(duplicated); break;
                case Code.Pop: if (!Pop(out _)) return false; break;
                case Code.Ldloc_0: case Code.Ldloc_1: case Code.Ldloc_2: case Code.Ldloc_3: case Code.Ldloc: case Code.Ldloc_S:
                {
                    int local = instruction.Operand is VariableDefinition variable ? variable.Index : (int)instruction.OpCode.Code - (int)Code.Ldloc_0;
                    if ((uint)local >= state.Locals.Length) return false; Push(state.Locals[local]); break;
                }
                case Code.Stloc_0: case Code.Stloc_1: case Code.Stloc_2: case Code.Stloc_3: case Code.Stloc: case Code.Stloc_S:
                {
                    int local = instruction.Operand is VariableDefinition variable ? variable.Index : (int)instruction.OpCode.Code - (int)Code.Stloc_0;
                    if ((uint)local >= state.Locals.Length || !Pop(out Value value)) return false; state.Locals[local] = value; break;
                }
                case Code.Ldfld:
                    if (!Pop(out Value fieldOwner) || !constructor && fieldOwner.Origin != Origin.Other) return false;
                    Push(new(Origin.Other, index)); break;
                case Code.Stfld:
                    if (!Pop(out Value fieldValue) || !Pop(out Value fieldReceiver) || fieldValue.Origin != Origin.Other ||
                        !constructor || fieldReceiver.Origin != Origin.Receiver) return false;
                    break;
                case Code.Call: case Code.Callvirt: case Code.Newobj:
                {
                    if (instruction.Operand is not MethodReference call) return false;
                    var arguments = new Value[call.Parameters.Count];
                    for (int position = arguments.Length - 1; position >= 0; position--)
                        if (!Pop(out arguments[position])) return false;
                    Value receiver = new(Origin.Other);
                    bool allocate = instruction.OpCode.Code == Code.Newobj;
                    if (!allocate && call.HasThis && !Pop(out receiver)) return false;
                    if (arguments.Any(value => value.Origin != Origin.Other)) return false;
                    if (allocate && !constructor && call.DeclaringType.FullName == method.ReturnType.FullName)
                    {
                        MethodDefinition? target = Resolve(call);
                        if (target is null || !PrivateConstructor(target, owner, dependencies, active)) return false;
                        Push(new(Origin.Fresh, index)); break;
                    }
                    if (receiver.Origin != Origin.Other)
                    {
                        if (receiver.Origin != (constructor ? Origin.Receiver : Origin.Fresh)) return false;
                        MethodDefinition? target = Resolve(call);
                        if (target is null) return false;
                        if (constructor && target.IsConstructor)
                        {
                            if (!PrivateConstructor(target, owner, dependencies, active)) return false;
                        }
                        else
                        {
                            FieldDefinition? field = Setter(target, owner);
                            if (field is null || arguments.Length != 1) return false;
                            dependencies.Add(DependencyGraph.Stable(target));
                            if (!constructor && field.FieldType.MetadataType == MetadataType.Boolean &&
                                arguments[0].Producer == index - 1 && Boolean(instructions[index - 1]) is bool value)
                            {
                                writes[index - 1] = new(field.Module.Assembly.Name.Name + ":" + field.FullName, index - 1, value);
                                state.AfterOwnedWrite = true;
                            }
                        }
                    }
                    else if (!constructor && state.AfterOwnedWrite) return false; // No opaque calls after an eligible write.
                    if (allocate || call.ReturnType.MetadataType != MetadataType.Void) Push(new(Origin.Other, index));
                    break;
                }
                case Code.Br: case Code.Br_S:
                    branches.Add(instructions.IndexOf((Instruction)instruction.Operand)); terminal = true; break;
                case Code.Brtrue: case Code.Brtrue_S: case Code.Brfalse: case Code.Brfalse_S:
                    if (!Pop(out _)) return false; branches.Add(instructions.IndexOf((Instruction)instruction.Operand)); break;
                case Code.Bgt: case Code.Bgt_S: case Code.Bgt_Un: case Code.Bgt_Un_S:
                case Code.Bge: case Code.Bge_S: case Code.Bge_Un: case Code.Bge_Un_S:
                case Code.Blt: case Code.Blt_S: case Code.Blt_Un: case Code.Blt_Un_S:
                case Code.Ble: case Code.Ble_S: case Code.Ble_Un: case Code.Ble_Un_S:
                case Code.Beq: case Code.Beq_S: case Code.Bne_Un: case Code.Bne_Un_S:
                    if (!Pop(out _) || !Pop(out _)) return false; branches.Add(instructions.IndexOf((Instruction)instruction.Operand)); break;
                case Code.Throw:
                    if (!Pop(out Value thrown) || thrown.Origin != Origin.Other) return false; terminal = true; break;
                case Code.Ret:
                    if (!constructor && (!Pop(out Value result) || result.Origin != Origin.Fresh)) return false;
                    if (state.Stack.Count != 0) return false; returned = true; terminal = true; break;
                default: return false;
            }
            if (!terminal) branches.Add(index + 1);
            foreach (int target in branches)
            {
                if (target <= index || target >= instructions.Count) return false;
                if (!states.TryGetValue(target, out State? previous)) { states.Add(target, state.Copy()); pending.Enqueue(target); continue; }
                if (previous.Stack.Count != state.Stack.Count) return false;
                bool changed = false;
                if (state.AfterOwnedWrite && !previous.AfterOwnedWrite) { previous.AfterOwnedWrite = true; changed = true; }
                for (int slot = 0; slot < previous.Stack.Count; slot++)
                {
                    Value merged = Merge(previous.Stack[slot], state.Stack[slot]);
                    changed |= merged != previous.Stack[slot]; previous.Stack[slot] = merged;
                }
                for (int slot = 0; slot < previous.Locals.Length; slot++)
                {
                    Value merged = Merge(previous.Locals[slot], state.Locals[slot]);
                    changed |= merged != previous.Locals[slot]; previous.Locals[slot] = merged;
                }
                if (changed) pending.Enqueue(target);
            }
        }
        return returned;
    }

    private static Value Merge(Value left, Value right) => new(left.Origin | right.Origin, left.Producer == right.Producer ? left.Producer : -1);
    private static bool SupportedBody(MethodDefinition method) => method.HasBody && !method.IsPInvokeImpl &&
        method.Body.Instructions.Count is > 0 and <= 256 && method.Body.ExceptionHandlers.Count == 0 && method.Body.Variables.Count <= 32;
    private static bool? Boolean(Instruction instruction) => instruction.OpCode.Code switch
    {
        Code.Ldc_I4_0 => false, Code.Ldc_I4_1 => true,
        Code.Ldc_I4 or Code.Ldc_I4_S when Convert.ToInt32(instruction.Operand) == 0 => false,
        Code.Ldc_I4 or Code.Ldc_I4_S when Convert.ToInt32(instruction.Operand) == 1 => true, _ => null
    };
    private static FieldDefinition? Setter(MethodDefinition method, TypeDefinition owner)
    {
        if (!SupportedBody(method) || method.IsStatic || method.Parameters.Count != 1 || method.ReturnType.MetadataType != MetadataType.Void ||
            method.IsVirtual && !method.IsFinal && method.DeclaringType != owner) return null;
        Instruction[] body = method.Body.Instructions.Where(instruction => instruction.OpCode.Code != Code.Nop).ToArray();
        if (body.Length != 4 || body[0].OpCode.Code != Code.Ldarg_0 || body[1].OpCode.Code != Code.Ldarg_1 ||
            body[2].OpCode.Code != Code.Stfld || body[3].OpCode.Code != Code.Ret || body[2].Operand is not FieldReference reference) return null;
        try
        {
            FieldDefinition? field = reference.Resolve();
            return field is not null && !field.IsStatic && field.DeclaringType == method.DeclaringType &&
                field.FieldType.FullName == method.Parameters[0].ParameterType.FullName ? field : null;
        }
        catch (AssemblyResolutionException) { return null; }
        catch (ResolutionException) { return null; }
    }
    private static bool HasFinalizer(TypeDefinition type)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        for (TypeDefinition? current = type; current is not null && current.FullName != "System.Object"; current = current.BaseType is null ? null : Resolve(current.BaseType))
        {
            if (!seen.Add(current.FullName) || current.Methods.Any(method => method.Name == "Finalize")) return true;
            if (current.BaseType is not null && Resolve(current.BaseType) is null) return true;
        }
        return false;
    }
    private static MethodDefinition? Resolve(MethodReference method)
    {
        try { return method.Resolve(); }
        catch (AssemblyResolutionException) { return null; }
        catch (ResolutionException) { return null; }
    }
    private static TypeDefinition? Resolve(TypeReference type)
    {
        try { return type.Resolve(); }
        catch (AssemblyResolutionException) { return null; }
        catch (ResolutionException) { return null; }
    }
    private static object? Operand(object? operand, MethodDefinition method) => operand switch
    {
        Instruction target => method.Body.Instructions.IndexOf(target),
        TypeReference type => type.FullName + "@" + type.Scope,
        MemberReference member => member.FullName + "@" + member.DeclaringType?.Scope,
        ParameterDefinition parameter => parameter.Index,
        VariableDefinition variable => variable.Index,
        _ => operand
    };
}
