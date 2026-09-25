using Mono.Cecil;

internal enum TaskReturnKind { Other, Task, Unresolved }

internal sealed class TaskTypeInspector
{
    private readonly Dictionary<string, TaskReturnKind> cache = new(StringComparer.Ordinal);

    public TaskReturnKind Classify(TypeReference type) => Classify(type, []);

    private TaskReturnKind Classify(TypeReference type, HashSet<GenericParameter> active)
    {
        if (type is GenericParameter parameter)
        {
            if (!active.Add(parameter)) return TaskReturnKind.Unresolved;
            try
            {
                TaskReturnKind[] constraints = parameter.Constraints.Select(constraint => Classify(constraint.ConstraintType, active)).ToArray();
                if (constraints.Contains(TaskReturnKind.Task)) return TaskReturnKind.Task;
                return constraints.Contains(TaskReturnKind.Unresolved) ? TaskReturnKind.Unresolved : TaskReturnKind.Other;
            }
            finally { active.Remove(parameter); }
        }
        if (type.IsByReference || type.IsPointer || type.IsArray || type.IsPrimitive || type.IsValueType || type.FullName == "System.Void")
            return TaskReturnKind.Other;
        string key = type.FullName + "@" + type.Scope;
        if (cache.TryGetValue(key, out TaskReturnKind cached)) return cached;
        var visited = new HashSet<string>(StringComparer.Ordinal);
        TypeReference? current = type;
        TaskReturnKind result = TaskReturnKind.Other;
        while (current is not null)
        {
            string name = current.GetElementType().FullName;
            if (name is "System.Threading.Tasks.Task" or "System.Threading.Tasks.Task`1") { result = TaskReturnKind.Task; break; }
            if (name == "System.Object") break;
            if (!visited.Add(current.FullName + "@" + current.Scope)) { result = TaskReturnKind.Unresolved; break; }
            try
            {
                TypeDefinition? definition = current.Resolve();
                if (definition is null) { result = TaskReturnKind.Unresolved; break; }
                current = definition.BaseType;
            }
            catch (AssemblyResolutionException) { result = TaskReturnKind.Unresolved; break; }
        }
        cache.Add(key, result);
        return result;
    }
}
