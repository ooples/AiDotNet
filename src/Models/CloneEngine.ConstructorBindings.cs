using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Models.Options;

namespace AiDotNet.Models;

public static partial class CloneEngine
{
    private static object ConstructFromBindings(Type type, ClonePlan plan, object source)
    {
        foreach (var candidate in plan.ConstructorBindings)
        {
            var originalArguments = new object?[candidate.Count];
            bool available = true;
            for (int i = 0; i < candidate.Count; i++)
            {
                if (candidate[i].Kind == CloneConstructorArgumentKind.ParameterDefault)
                    originalArguments[i] = Type.Missing;
                else if (!TryReadBoundMember(candidate[i], source, out originalArguments[i]))
                {
                    available = false;
                    break;
                }
            }
            if (!available) continue;

            var constructors = type.GetConstructors(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                .Where(constructor => MatchesBoundConstructor(constructor, candidate, originalArguments)).ToArray();
            if (constructors.Length == 0) continue;
            if (constructors.Length != 1)
                throw new InvalidOperationException($"The explicit clone bindings for {type.Name} match multiple constructors.");

            var arguments = new object?[originalArguments.Length];
            for (int i = 0; i < arguments.Length; i++)
                arguments[i] = ReferenceEquals(originalArguments[i], Type.Missing)
                    ? Type.Missing : DuplicateConstructorArgument(originalArguments[i]);

            return constructors[0].Invoke(BindingFlags.OptionalParamBinding, binder: null, arguments, culture: null);
        }

        // An explicit owner is configuration evidence. Falling back to defaults after it fails
        // discards supplied components and can allocate a completely different model architecture.
        throw new InvalidOperationException($"No constructor of {type.Name} accepts its explicitly bound configuration.");
    }

    private static bool MatchesBoundConstructor(ConstructorInfo constructor,
        IReadOnlyList<CloneConstructorArgumentBinding> bindings, object?[] arguments)
    {
        var parameters = constructor.GetParameters();
        if (parameters.Length != bindings.Count) return false;
        for (int i = 0; i < parameters.Length; i++)
        {
            if (!string.Equals(parameters[i].Name, bindings[i].ParameterName, StringComparison.Ordinal)) return false;
            if (bindings[i].Kind == CloneConstructorArgumentKind.ParameterDefault)
            {
                if (!parameters[i].HasDefaultValue) return false;
            }
            else if (arguments[i] is null)
            {
                if (!parameters[i].HasDefaultValue || parameters[i].ParameterType.IsValueType
                    && Nullable.GetUnderlyingType(parameters[i].ParameterType) is null) return false;
            }
            else if (!parameters[i].ParameterType.IsInstanceOfType(arguments[i])) return false;
        }
        return true;
    }

    private static bool TryReadBoundMember(CloneConstructorArgumentBinding binding, object source, out object? value)
    {
        object? current = source;
        var genericArguments = new Dictionary<Type, Type>();
        foreach (var declaredMember in binding.Members)
        {
            if (current is null)
            {
                // A native instance may not own the nested state required by a separate ONNX or
                // alternate-provider constructor. That candidate is unavailable, not malformed.
                value = null;
                return false;
            }
            var declaringType = declaredMember.DeclaringType
                ?? throw new InvalidOperationException("An explicit clone member has no declaring type.");
            Type? closedOwner = current.GetType();
            while (closedOwner is not null && !MatchesDeclaringOwner(closedOwner, declaringType, genericArguments))
                closedOwner = closedOwner.BaseType;
            if (closedOwner is null)
                throw new InvalidOperationException($"The explicit clone owner {declaringType} for '{binding.ParameterName}' is absent from {current.GetType()}.");

            const BindingFlags flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.DeclaredOnly;
            current = declaredMember switch
            {
                FieldInfo => (closedOwner.GetField(declaredMember.Name, flags)
                    ?? throw new InvalidOperationException($"Missing explicitly bound field {closedOwner}.{declaredMember.Name}.")).GetValue(current),
                PropertyInfo => (closedOwner.GetProperty(declaredMember.Name, flags)
                    ?? throw new InvalidOperationException($"Missing explicitly bound property {closedOwner}.{declaredMember.Name}.")).GetValue(current),
                _ => throw new InvalidOperationException("An explicit clone path contains an unsupported member kind.")
            };
        }
        value = current;
        return true;
    }

    private static bool MatchesDeclaringOwner(Type actual, Type declared, Dictionary<Type, Type> arguments)
    {
        if (actual == declared) return true;
        if (declared.IsGenericParameter)
        {
            if (arguments.TryGetValue(declared, out var previous)) return actual == previous;
            arguments.Add(declared, actual);
            return true;
        }
        if (declared.IsArray)
        {
            if (!actual.IsArray || actual.GetArrayRank() != declared.GetArrayRank()) return false;
            var actualElement = actual.GetElementType();
            var declaredElement = declared.GetElementType();
            if (actualElement is null || declaredElement is null) return false;
            // A rank-one vector and a rank-one multidimensional array have different types.
            if ((actual == actualElement.MakeArrayType()) != (declared == declaredElement.MakeArrayType())) return false;
            return MatchesDeclaringOwner(actualElement, declaredElement, arguments);
        }
        if (!declared.IsGenericType || !actual.IsGenericType
            || actual.GetGenericTypeDefinition() != declared.GetGenericTypeDefinition()) return false;
        var actualArguments = actual.GetGenericArguments();
        var declaredArguments = declared.GetGenericArguments();
        for (int i = 0; i < actualArguments.Length; i++)
            if (!MatchesDeclaringOwner(actualArguments[i], declaredArguments[i], arguments)) return false;
        return true;
    }

    private static void RestoreBoundConstructorConfiguration(object source, object destination, ClonePlan plan)
    {
        var restored = new HashSet<object>(ReferenceIdentityComparer.Instance);
        foreach (var candidate in plan.ConstructorBindings)
        {
            foreach (var binding in candidate)
            {
                if (binding.Kind == CloneConstructorArgumentKind.ParameterDefault) continue;
                if (!TryReadBoundMember(binding, source, out var sourceValue) || sourceValue is not ModelOptions sourceOptions) continue;
                if (!TryReadBoundMember(binding, destination, out var destinationValue) || destinationValue is not ModelOptions destinationOptions
                    || destinationOptions.GetType() != sourceOptions.GetType() || !restored.Add(destinationOptions)) continue;
                var optionPlan = CloneRegistry.GetPlan(sourceOptions.GetType());
                var pending = new List<(ClonePlanEntry Entry, object? Value)>(optionPlan.Entries.Count);
                foreach (var entry in optionPlan.Entries)
                    pending.Add((entry, entry.Property.GetValue(sourceOptions)));
                Assign(sourceOptions.GetType(), destinationOptions, pending);
            }
        }
    }
}
