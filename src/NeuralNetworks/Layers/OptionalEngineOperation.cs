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
        var call = Expression.Call(parameters[0], method, parameters.Skip(1));
        return Expression.Lambda<TDelegate>(call, parameters).Compile();
    }
}
