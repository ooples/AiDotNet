using System.Reflection;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>Optional CSR attention operations provided together by newer Tensors engines.</summary>
internal sealed class CsrAttentionOperations<T>
{
    internal static CsrAttentionOperations<T>? Available { get; } = Bind();

    internal Func<Tensor<T>, object?> CreatePattern { get; }
    internal Func<IEngine, Tensor<T>, Tensor<T>, object, double, double, Random?, Tensor<T>> Aggregate { get; }

    private CsrAttentionOperations(Func<Tensor<T>, object?> createPattern,
        Func<IEngine, Tensor<T>, Tensor<T>, object, double, double, Random?, Tensor<T>> aggregate)
    {
        CreatePattern = createPattern;
        Aggregate = aggregate;
    }

    private static CsrAttentionOperations<T>? Bind()
    {
        // The published 0.151.2 package predates the CSR type and engine operation. Discover the
        // optional capability once per element type; forwards then use ordinary typed delegates.
        Type? pattern = typeof(IEngine).Assembly.GetType("AiDotNet.Tensors.Engines.CsrGraph");
        if (pattern is null) return null;
        var binder = typeof(CsrAttentionOperations<T>).GetMethod(nameof(BindTyped), BindingFlags.Static | BindingFlags.NonPublic)!;
        try
        {
            var bind = (Func<CsrAttentionOperations<T>?>)binder.MakeGenericMethod(pattern)
                .CreateDelegate(typeof(Func<CsrAttentionOperations<T>?>));
            return bind();
        }
        catch (NotSupportedException) { return null; }
        catch (ArgumentException) { return null; }
    }

    private static CsrAttentionOperations<T>? BindTyped<TPattern>() where TPattern : class
    {
        var fromDense = typeof(TPattern).GetMethod("FromDenseAdjacency", BindingFlags.Static | BindingFlags.Public);
        var allRows = typeof(TPattern).GetProperty("EveryRowHasEdge")?.GetMethod;
        var operation = typeof(IEngine).GetMethods().FirstOrDefault(m => m.Name == "GraphAttentionCsr"
            && m.IsGenericMethodDefinition && m.GetParameters().Length == 6 && m.GetParameters()[2].ParameterType == typeof(TPattern));
        if (fromDense is null || !fromDense.IsGenericMethodDefinition || allRows is null || operation is null) return null;

        var create = (Func<Tensor<T>, TPattern?>)fromDense.MakeGenericMethod(typeof(T)).CreateDelegate(typeof(Func<Tensor<T>, TPattern?>));
        var everyRowHasEdge = (Func<TPattern, bool>)allRows.CreateDelegate(typeof(Func<TPattern, bool>));
        var aggregate = (Func<IEngine, Tensor<T>, Tensor<T>, TPattern, double, double, Random?, Tensor<T>>)
            operation.MakeGenericMethod(typeof(T)).CreateDelegate(typeof(Func<IEngine, Tensor<T>, Tensor<T>, TPattern, double, double, Random?, Tensor<T>>));
        return new CsrAttentionOperations<T>(adjacency =>
        {
            var pattern = create(adjacency);
            return pattern is not null && everyRowHasEdge(pattern) ? pattern : null;
        }, (engine, transformed, attention, pattern, slope, dropout, random) =>
            aggregate(engine, transformed, attention, (TPattern)pattern, slope, dropout, random));
    }
}
