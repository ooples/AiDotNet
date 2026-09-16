using Mono.Cecil;
using Mono.Cecil.Cil;

internal enum OpenDependencyKind { ExternalCall, VirtualDispatch, IndirectCall, NativeCode, UnresolvedLocalCall, ExternalField, UnresolvedLocalField }
internal sealed record OpenDependency(OpenDependencyKind Kind, string Target);
internal sealed record MethodDependencyNode(string Key, string Name, string[] LocalCalls,
    string[] StaticFields, OpenDependency[] OpenDependencies);
internal sealed record MethodDependencyGraph(int Schema, string InputHash, MethodDependencyNode[] Methods);

// Evidence for a later conservative planner, not a completeness certificate.
// A visited method's unexecuted branches still contribute static call edges.
// Unresolved/virtual/external paths stay explicit rather than silently disappearing.
internal static class DependencyGraph
{
    public static MethodDependencyGraph Read(AssemblyDefinition assembly, string inputHash)
    {
        var nodes = new List<MethodDependencyNode>();
        foreach (TypeDefinition type in Types(assembly.MainModule.Types))
        foreach (MethodDefinition method in type.Methods)
        {
            if (!method.HasBody && !method.IsPInvokeImpl) continue;
            var local = new HashSet<string>(StringComparer.Ordinal);
            var fields = new HashSet<string>(StringComparer.Ordinal);
            var open = new HashSet<OpenDependency>();
            // Type initialization can execute without an explicit IL call.
            AddTypeInitializer(type, local, inputHash);
            if (method.IsPInvokeImpl) open.Add(new(OpenDependencyKind.NativeCode, method.FullName));
            if (method.HasBody)
            foreach (Instruction instruction in method.Body.Instructions)
            {
                if (instruction.OpCode.Code == Code.Calli)
                    open.Add(new(OpenDependencyKind.IndirectCall, method.FullName));
                if (instruction.Operand is FieldReference field)
                {
                    if (field.DeclaringType.Scope != assembly.MainModule)
                        open.Add(new(OpenDependencyKind.ExternalField, field.FullName));
                    else
                    {
                        FieldDefinition? definition = field.Resolve();
                        if (definition is null) open.Add(new(OpenDependencyKind.UnresolvedLocalField, field.FullName));
                        else if (definition.IsStatic && !definition.IsLiteral)
                        {
                            // readonly references may refer to mutable shared state.
                            fields.Add(definition.FullName);
                            AddTypeInitializer(definition.DeclaringType, local, inputHash);
                        }
                    }
                }
                if (instruction.Operand is not MethodReference call) continue;
                if (call.DeclaringType.Scope != assembly.MainModule)
                {
                    open.Add(new(OpenDependencyKind.ExternalCall, call.FullName));
                    continue;
                }
                MethodDefinition? target = call.Resolve();
                if (target is null)
                {
                    open.Add(new(OpenDependencyKind.UnresolvedLocalCall, call.FullName));
                    continue;
                }
                local.Add(Key(target, inputHash));
                if (target.IsVirtual && !target.IsFinal && !target.DeclaringType.IsSealed)
                    open.Add(new(OpenDependencyKind.VirtualDispatch, target.FullName));
            }
            foreach (CustomAttribute attribute in method.CustomAttributes)
            {
                if (attribute.AttributeType.FullName is not ("System.Runtime.CompilerServices.AsyncStateMachineAttribute" or
                    "System.Runtime.CompilerServices.IteratorStateMachineAttribute" or "System.Runtime.CompilerServices.AsyncIteratorStateMachineAttribute")) continue;
                if (attribute.ConstructorArguments.Count != 1 || attribute.ConstructorArguments[0].Value is not TypeReference machine)
                    throw new InvalidDataException("Malformed state-machine metadata.");
                TypeDefinition stateType = machine.Resolve();
                foreach (MethodDefinition member in stateType.Methods.Where(member => member.HasBody))
                    local.Add(Key(member, inputHash));
            }
            nodes.Add(new(Key(method, inputHash), method.FullName, local.Order(StringComparer.Ordinal).ToArray(),
                fields.Order(StringComparer.Ordinal).ToArray(), open.OrderBy(item => item.Kind).ThenBy(item => item.Target, StringComparer.Ordinal).ToArray()));
        }
        return new(1, inputHash, nodes.OrderBy(node => node.Key, StringComparer.Ordinal).ToArray());
    }

    private static string Key(MethodDefinition method, string inputHash) => $"{inputHash}:{method.MetadataToken.ToInt32():X8}";

    private static void AddTypeInitializer(TypeDefinition type, HashSet<string> calls, string hash)
    {
        foreach (MethodDefinition initializer in type.Methods.Where(method => method.IsConstructor && method.IsStatic && method.HasBody))
            calls.Add(Key(initializer, hash));
    }

    private static IEnumerable<TypeDefinition> Types(IEnumerable<TypeDefinition> roots)
    {
        foreach (TypeDefinition type in roots)
        {
            yield return type;
            foreach (TypeDefinition nested in Types(type.NestedTypes)) yield return nested;
        }
    }
}
