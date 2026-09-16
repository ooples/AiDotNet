using Mono.Cecil;
using Mono.Cecil.Cil;

internal enum OpenDependencyKind { ExternalCall, VirtualDispatch, IndirectCall, NativeCode, UnresolvedLocalCall, ExternalField, UnresolvedLocalField, UnboundType }
internal sealed record OpenDependency(OpenDependencyKind Kind, string Target);
internal sealed record MethodDependencyNode(string Key, string Name, string[] LocalCalls,
    string[] StaticFields, OpenDependency[] OpenDependencies);
internal sealed record MethodDependencyGraph(int Schema, string InputHash, MethodDependencyNode[] Methods);

// Evidence for a later conservative planner, not a completeness certificate.
// A visited method's unexecuted branches still contribute static call edges.
// Unresolved/virtual/external paths stay explicit rather than silently disappearing.
internal static class DependencyGraph
{
    public static MethodDependencyGraph Read(AssemblyDefinition assembly, string inputHash, IReadOnlySet<string>? linkedAssemblyPaths = null,
        ManagedDependencyReader? managed = null, IEnumerable<MethodDefinition>? selectedMethods = null)
    {
        var nodes = new List<MethodDependencyNode>();
        foreach (MethodDefinition method in selectedMethods ?? Types(assembly.MainModule.Types).SelectMany(type => type.Methods)
                     .Where(method => method.HasBody || method.IsPInvokeImpl))
        {
            TypeDefinition type = method.DeclaringType;
            managed?.BeginMethod();
            var local = new HashSet<string>(StringComparer.Ordinal);
            var fields = new HashSet<string>(StringComparer.Ordinal);
            var open = new HashSet<OpenDependency>();
            if (managed is not null && (!managed.IsBound(method.ReturnType) || method.Parameters.Any(parameter => !managed.IsBound(parameter.ParameterType)) ||
                method.HasBody && method.Body.Variables.Any(variable => !managed.IsBound(variable.VariableType))))
                open.Add(new(OpenDependencyKind.UnboundType, method.FullName));
            // Type initialization can execute without an explicit IL call.
            AddTypeInitializer(type, local, inputHash, managed);
            // Dependency modules can initialize before their first type is
            // touched too. Their entry points are not test-assembly fixtures.
            foreach (TypeDefinition moduleType in method.Module.Types.Where(candidate => candidate.Name == "<Module>"))
                AddTypeInitializer(moduleType, local, inputHash, managed);
            if (method.IsPInvokeImpl) open.Add(new(OpenDependencyKind.NativeCode, method.FullName));
            else if (!method.HasBody) open.Add(new(method.IsAbstract ? OpenDependencyKind.VirtualDispatch : OpenDependencyKind.NativeCode, method.FullName));
            if (method.HasBody)
            foreach (Instruction instruction in method.Body.Instructions)
            {
                if (instruction.OpCode.Code == Code.Calli)
                    open.Add(new(OpenDependencyKind.IndirectCall, method.FullName));
                if (instruction.OpCode.Code is Code.Localloc or Code.Cpblk or Code.Initblk)
                    open.Add(new(OpenDependencyKind.NativeCode, method.FullName));
                if (managed is not null && instruction.Operand is TypeReference operandType && !managed.IsBound(operandType))
                    open.Add(new(OpenDependencyKind.UnboundType, operandType.FullName));
                if (instruction.Operand is FieldReference field)
                {
                    if (managed is not null && (!managed.IsBound(field.DeclaringType) || !managed.IsBound(field.FieldType)))
                        open.Add(new(OpenDependencyKind.UnboundType, field.FullName));
                    if (field.DeclaringType.Scope != assembly.MainModule)
                    {
                        FieldDefinition? linked = ResolveLinked(field, linkedAssemblyPaths) ?? managed?.Resolve(field);
                        if (linked is null) open.Add(new(OpenDependencyKind.ExternalField, field.FullName));
                        else if (linked.IsStatic && !linked.IsLiteral)
                        {
                            fields.Add(linked.Module.Assembly.Name.Name + ":" + linked.FullName);
                            AddLinkedInitializer(linked.DeclaringType, local, managed);
                        }
                    }
                    else
                    {
                        FieldDefinition? definition = field.Resolve();
                        if (definition is null) open.Add(new(OpenDependencyKind.UnresolvedLocalField, field.FullName));
                        else if (definition.IsStatic && !definition.IsLiteral)
                        {
                            // readonly references may refer to mutable shared state.
                            fields.Add(assembly.Name.Name + ":" + definition.FullName);
                            AddTypeInitializer(definition.DeclaringType, local, inputHash, managed);
                        }
                    }
                }
                if (instruction.Operand is not MethodReference call) continue;
                if (managed is not null && (!managed.IsBound(call.DeclaringType) || !managed.IsBound(call.ReturnType) ||
                    call.Parameters.Any(parameter => !managed.IsBound(parameter.ParameterType)) ||
                    call is GenericInstanceMethod generic && generic.GenericArguments.Any(argument => !managed.IsBound(argument))))
                    open.Add(new(OpenDependencyKind.UnboundType, call.FullName));
                if (call.DeclaringType.Scope != assembly.MainModule)
                {
                    MethodDefinition? linked = ResolveLinked(call, linkedAssemblyPaths) ?? managed?.Resolve(call);
                    if (linked is null) open.Add(new(OpenDependencyKind.ExternalCall, call.FullName));
                    else
                    {
                        local.Add(Stable(linked));
                        managed?.Observe(linked);
                        AddLinkedInitializer(linked.DeclaringType, local, managed);
                        if (instruction.OpCode.Code is Code.Callvirt or Code.Ldvirtftn && linked.IsVirtual && !linked.IsFinal && !linked.DeclaringType.IsSealed)
                            open.Add(new(OpenDependencyKind.VirtualDispatch, linked.FullName));
                    }
                    continue;
                }
                MethodDefinition? target = call.Resolve();
                if (target is null)
                {
                    open.Add(new(OpenDependencyKind.UnresolvedLocalCall, call.FullName));
                    continue;
                }
                local.Add(Key(target, inputHash));
                managed?.Observe(target);
                if (instruction.OpCode.Code is Code.Callvirt or Code.Ldvirtftn && target.IsVirtual && !target.IsFinal && !target.DeclaringType.IsSealed)
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
                {
                    local.Add(Key(member, inputHash));
                    managed?.Observe(member);
                }
            }
            managed?.EndMethod(open.Count == 0);
            nodes.Add(new(Key(method, inputHash), method.FullName, local.Order(StringComparer.Ordinal).ToArray(),
                fields.Order(StringComparer.Ordinal).ToArray(), open.OrderBy(item => item.Kind).ThenBy(item => item.Target, StringComparer.Ordinal).ToArray()));
        }
        return new(1, inputHash, nodes.OrderBy(node => node.Key, StringComparer.Ordinal).ToArray());
    }

    private static string Key(MethodDefinition method, string inputHash) => $"{inputHash}:{method.MetadataToken.ToInt32():X8}";

    // Cecil FullName omits method generic arity. CoreLib has legitimate
    // overloads whose rendered parameter names coincide but arities differ.
    internal static string Stable(MethodDefinition method) => method.Module.Assembly.Name.Name + ":" + method.FullName +
        (method.HasGenericParameters ? "#arity=" + method.GenericParameters.Count.ToString(System.Globalization.CultureInfo.InvariantCulture) : "");

    private static MethodDefinition? ResolveLinked(MethodReference reference, IReadOnlySet<string>? paths)
    {
        if (paths is null) return null;
        try
        {
            MethodDefinition? target = reference.Resolve();
            return target is not null && paths.Contains(Path.GetFullPath(target.Module.FileName)) ? target : null;
        }
        catch (AssemblyResolutionException) { return null; }
    }

    private static FieldDefinition? ResolveLinked(FieldReference reference, IReadOnlySet<string>? paths)
    {
        if (paths is null) return null;
        try
        {
            FieldDefinition? target = reference.Resolve();
            return target is not null && paths.Contains(Path.GetFullPath(target.Module.FileName)) ? target : null;
        }
        catch (AssemblyResolutionException) { return null; }
    }

    private static void AddLinkedInitializer(TypeDefinition type, HashSet<string> calls, ManagedDependencyReader? managed)
    {
        foreach (MethodDefinition initializer in type.Methods.Where(method => method.IsConstructor && method.IsStatic && method.HasBody))
        {
            calls.Add(Stable(initializer));
            managed?.Observe(initializer);
        }
    }

    private static void AddTypeInitializer(TypeDefinition type, HashSet<string> calls, string hash, ManagedDependencyReader? managed)
    {
        foreach (MethodDefinition initializer in type.Methods.Where(method => method.IsConstructor && method.IsStatic && method.HasBody))
        {
            calls.Add(Key(initializer, hash));
            managed?.Observe(initializer);
        }
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
