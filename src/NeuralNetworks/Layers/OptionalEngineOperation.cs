using System.Linq.Expressions;
using System.Reflection;

namespace AiDotNet.NeuralNetworks.Layers;

internal static class OptionalEngineOperation
{
    // The CLR cannot create an open delegate to a generic virtual interface member. Emit its
    // ordinary interface call once so each forward dispatches to the engine that owns the tensor.
    internal static TDelegate Bind<TDelegate>(MethodInfo method) where TDelegate : Delegate
    {
        var invoke = typeof(TDelegate).GetMethod("Invoke")!;
        var parameters = invoke.GetParameters().Select(p => Expression.Parameter(p.ParameterType, p.Name)).ToArray();
        var call = method.IsStatic ? Expression.Call(method, parameters) : Expression.Call(parameters[0], method, parameters.Skip(1));
        return Expression.Lambda<TDelegate>(call, parameters).Compile();
    }

    internal static TDelegate? Find<TDelegate>(Type? owner, string name, Type? elementType = null) where TDelegate : Delegate
    {
        if (owner is null) return null;
        var signature = typeof(TDelegate).GetMethod("Invoke")!;
        var parameters = signature.GetParameters().Select(p => p.ParameterType).ToArray();
        const BindingFlags flags = BindingFlags.Static | BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        foreach (var candidate in owner.GetMethods(flags))
        {
            if (candidate.Name != name || candidate.IsGenericMethodDefinition != (elementType is not null)) continue;
            if (candidate.IsGenericMethodDefinition && candidate.GetGenericArguments().Length != 1) continue;
            try
            {
                var method = elementType is null ? candidate : candidate.MakeGenericMethod(elementType);
                if (method.ReturnType != signature.ReturnType || (!method.IsStatic && parameters.FirstOrDefault() != owner)) continue;
                if (!method.GetParameters().Select(p => p.ParameterType).SequenceEqual(parameters.Skip(method.IsStatic ? 0 : 1))) continue;
                return Bind<TDelegate>(method);
            }
            catch (ArgumentException) { }
            catch (NotSupportedException) { }
        }
        return null;
    }
}
