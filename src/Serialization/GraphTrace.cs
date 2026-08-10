using System.Globalization;
using System.Reflection;
using System.Text;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serialization;

/// <summary>
/// Records what a traceable expression computes, as a graph, and replays it.
/// </summary>
/// <remarks>
/// <para>
/// This is the strongest of the three descriptions <see cref="DelegateState"/> tries, and the only
/// one that survives a closure. A delegate built over captured state has no name to refer to, but
/// running it once over autodiff nodes leaves behind a record of the operations it performed, and
/// that record is data.
/// </para>
/// <para>
/// It works because <c>TensorOperations</c> already tags every node it produces with an
/// <see cref="OperationType"/> and, for the operations that take them, an
/// <c>OperationParams</c> dictionary whose keys match the method's parameter names. Nothing
/// consumed either before this; they were written for a JIT that had not been built.
/// </para>
/// <para>
/// Replay resolves operations by name against <c>TensorOperations&lt;T&gt;</c> and nothing else, so
/// a saved graph can only ever invoke a tensor operation. That is the security property Keras's
/// Lambda layer gives up by marshalling bytecode, and it comes from construction here rather than
/// from a filter that has to be kept ahead of attackers.
/// </para>
/// </remarks>
public static class GraphTrace
{
    private const string InputOp = "Input";

    /// <summary>
    /// Runs <paramref name="expression"/> once over autodiff nodes and records the operations it
    /// performed.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="expression">The traceable expression the layer was built with.</param>
    /// <param name="inputShape">The layer's input shape, used to make a probe tensor.</param>
    /// <returns>The recorded graph, or <c>null</c> when it cannot be recorded faithfully.</returns>
    /// <remarks>
    /// Returns <c>null</c> rather than a partial graph whenever anything is not fully recoverable:
    /// an operation with no tag, a parameter value of a type that will not round-trip, or an
    /// expression that throws on the probe. The caller then falls back to a weaker description,
    /// which is the point of the tiers.
    /// </remarks>
    public static string? Trace<T>(Func<ComputationNode<T>, ComputationNode<T>>? expression, int[]? inputShape)
    {
        if (expression is null) return null;

        ComputationNode<T> output;
        ComputationNode<T> input;
        try
        {
            // A lazy or batch axis is recorded as 0 or -1; the probe only has to have the right
            // rank for the expression to run, so those become 1.
            var probeShape = (inputShape is null || inputShape.Length == 0 ? [1] : inputShape)
                .Select(d => d > 0 ? d : 1).ToArray();

            input = TensorOperations<T>.Variable(new Tensor<T>(probeShape));
            output = expression(input);
        }
        catch (Exception)
        {
            // An expression that will not run on a probe cannot be recorded. That is a fallback,
            // not a failure.
            return null;
        }

        var order = new List<ComputationNode<T>>();
        var ids = new Dictionary<ComputationNode<T>, int>(ReferenceComparer<T>.Instance);
        if (!Order(output, order, ids, input)) return null;

        var sb = new StringBuilder();
        foreach (var node in order)
        {
            if (sb.Length > 0) sb.Append(';');
            sb.Append(ids[node]).Append('=');

            if (ReferenceEquals(node, input))
            {
                sb.Append(InputOp);
                continue;
            }

            sb.Append(node.OperationType!.Value.ToString());
            sb.Append('(');
            for (var i = 0; i < node.Parents.Count; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append(ids[node.Parents[i]]);
            }
            sb.Append(')');

            var encoded = EncodeParams(node.OperationParams);
            if (encoded is null) return null;
            sb.Append(encoded);
        }

        // Separated, not glued: appending ">;" straight onto the last node left it reading
        // "2=Square(1)>". That parsed only because the stray ">" landed in the parameter tail,
        // which is ignored -- the same shape in ExpressionState hit int.Parse and threw.
        sb.Append(';').Append(ids[output]);
        return sb.ToString();
    }

    /// <summary>Depth-first post-order, rejecting anything that cannot be replayed.</summary>
    private static bool Order<T>(
        ComputationNode<T> node,
        List<ComputationNode<T>> order,
        Dictionary<ComputationNode<T>, int> ids,
        ComputationNode<T> input)
    {
        if (ids.ContainsKey(node)) return true;

        if (!ReferenceEquals(node, input))
        {
            // A leaf that is not the input is a captured constant or weight. Its value is not
            // construction state and may be arbitrarily large, so the graph declines to carry it.
            if (node.Parents is null || node.Parents.Count == 0) return false;
            if (node.OperationType is null) return false;

            foreach (var parent in node.Parents)
            {
                if (!Order(parent, order, ids, input)) return false;
            }
        }

        ids[node] = ids.Count;
        order.Add(node);
        return true;
    }

    private static string? EncodeParams(Dictionary<string, object>? parameters)
    {
        if (parameters is null || parameters.Count == 0) return string.Empty;

        var sb = new StringBuilder("{");
        var first = true;
        foreach (var kvp in parameters.OrderBy(k => k.Key, StringComparer.Ordinal))
        {
            var encoded = EncodeValue(kvp.Value);
            if (encoded is null) return null;

            if (!first) sb.Append(',');
            first = false;
            sb.Append(kvp.Key).Append('=').Append(encoded);
        }

        return sb.Append('}').ToString();
    }

    private static string? EncodeValue(object? value) => value switch
    {
        null => "~",
        bool b => b ? "true" : "false",
        int i => i.ToString(CultureInfo.InvariantCulture),
        long l => l.ToString(CultureInfo.InvariantCulture),
        double d => d.ToString("R", CultureInfo.InvariantCulture),
        float f => f.ToString("R", CultureInfo.InvariantCulture),
        string s => s.IndexOfAny([',', '}', '=', ';']) >= 0 ? null : "'" + s,
        int[] a => "[" + string.Join("|", a) + "]",
        _ => null,
    };

    /// <summary>Rebuilds the expression a graph records.</summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="graph">A graph produced by <see cref="Trace"/>.</param>
    /// <param name="layerName">The layer being rebuilt, named in any failure.</param>
    /// <param name="key">The constructor parameter, named in any failure.</param>
    /// <returns>The replayed expression.</returns>
    public static Func<ComputationNode<T>, ComputationNode<T>> Compile<T>(string graph, string layerName, string key)
    {
        var parts = graph.Split(';');
        if (parts.Length < 2 || !int.TryParse(parts[^1], NumberStyles.Integer, CultureInfo.InvariantCulture, out var outputId))
            throw Bad(layerName, key, "its recorded graph has no output node.");

        // Take rather than a range: an array range needs RuntimeHelpers.GetSubArray, which net471
        // does not have. String ranges below are fine -- those compile to Substring.
        var steps = parts.Take(parts.Length - 1).Where(p => p.Length > 0 && p != ">").ToArray();

        return input =>
        {
            var nodes = new Dictionary<int, ComputationNode<T>>();
            foreach (var step in steps)
            {
                var eq = step.IndexOf('=');
                if (eq < 0 || !int.TryParse(step[..eq], out var id))
                    throw Bad(layerName, key, $"a recorded step '{step}' is malformed.");

                var body = step[(eq + 1)..];
                if (body == InputOp)
                {
                    nodes[id] = input;
                    continue;
                }

                nodes[id] = Replay<T>(body, nodes, layerName, key);
            }

            if (!nodes.TryGetValue(outputId, out var output))
                throw Bad(layerName, key, "its recorded output node was never produced.");

            return output;
        };
    }

    private static ComputationNode<T> Replay<T>(
        string body, Dictionary<int, ComputationNode<T>> nodes, string layerName, string key)
    {
        var open = body.IndexOf('(');
        var close = body.IndexOf(')');
        if (open < 0 || close < open) throw Bad(layerName, key, $"a recorded step '{body}' is malformed.");

        var opName = body[..open];
        var argIds = body[(open + 1)..close];
        var parents = argIds.Length == 0
            ? []
            : argIds.Split(',').Select(s => nodes[int.Parse(s, CultureInfo.InvariantCulture)]).ToArray();

        var parameters = DecodeParams(body[(close + 1)..]);

        // Resolved against TensorOperations<T> and nothing else, so a saved graph cannot name any
        // other method. The allowlist is the lookup, not a filter layered over it.
        var ops = typeof(TensorOperations<>).MakeGenericType(typeof(T));
        var candidates = ops.GetMethods(BindingFlags.Public | BindingFlags.Static)
            .Where(m => m.Name == opName && m.ReturnType == typeof(ComputationNode<T>))
            .ToArray();

        foreach (var method in candidates)
        {
            var bound = Bind<T>(method, parents, parameters);
            if (bound is not null) return (ComputationNode<T>)method.Invoke(null, bound)!;
        }

        throw Bad(layerName, key,
            $"its recorded graph uses '{opName}' with {parents.Length} input(s), which TensorOperations<{typeof(T).Name}> "
            + "no longer provides in a matching form.");
    }

    /// <summary>
    /// Fills a method's parameters from the graph: node parameters in order from the recorded
    /// parents, everything else by name from the recorded values.
    /// </summary>
    private static object?[]? Bind<T>(MethodInfo method, ComputationNode<T>[] parents, Dictionary<string, string> parameters)
    {
        var formal = method.GetParameters();
        var args = new object?[formal.Length];
        var nextParent = 0;

        foreach (var (p, i) in formal.Select((p, i) => (p, i)))
        {
            if (p.ParameterType == typeof(ComputationNode<T>))
            {
                if (nextParent >= parents.Length) return null;
                args[i] = parents[nextParent++];
                continue;
            }

            if (parameters.TryGetValue(p.Name ?? string.Empty, out var raw))
            {
                var value = DecodeValue(raw, p.ParameterType);
                if (value is null && raw != "~") return null;
                args[i] = value;
                continue;
            }

            if (p.HasDefaultValue) { args[i] = p.DefaultValue; continue; }

            return null;
        }

        return nextParent == parents.Length ? args : null;
    }

    private static Dictionary<string, string> DecodeParams(string tail)
    {
        // Case-insensitive because the recorded keys are PascalCase ("Axis") while the method
        // parameters they correspond to are camelCase ("axis").
        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var open = tail.IndexOf('{');
        var close = tail.LastIndexOf('}');
        if (open < 0 || close <= open) return result;

        foreach (var pair in tail[(open + 1)..close].Split(','))
        {
            var eq = pair.IndexOf('=');
            if (eq > 0) result[pair[..eq]] = pair[(eq + 1)..];
        }

        return result;
    }

    private static object? DecodeValue(string raw, Type target)
    {
        if (raw == "~") return null;
        if (raw.StartsWith("'", StringComparison.Ordinal)) return raw[1..];

        if (raw.StartsWith("[", StringComparison.Ordinal) && raw.EndsWith("]", StringComparison.Ordinal))
        {
            var inner = raw[1..^1];
            return inner.Length == 0
                ? []
                : inner.Split('|').Select(s => int.Parse(s, CultureInfo.InvariantCulture)).ToArray();
        }

        var type = Nullable.GetUnderlyingType(target) ?? target;
        if (type == typeof(bool)) return raw == "true";
        if (type.IsEnum) return int.TryParse(raw, out var e) ? Enum.ToObject(type, e) : Enum.Parse(type, raw, true);

        try
        {
            return Convert.ChangeType(raw, type, CultureInfo.InvariantCulture);
        }
        catch (Exception)
        {
            return null;
        }
    }

    private static InvalidOperationException Bad(string layerName, string key, string why)
        => new($"Cannot rebuild {layerName}: constructor parameter '{key}' was saved as a traced graph, but {why}");

    /// <summary>Nodes are identified by reference; two distinct nodes may hold equal tensors.</summary>
    private sealed class ReferenceComparer<T> : IEqualityComparer<ComputationNode<T>>
    {
        internal static readonly ReferenceComparer<T> Instance = new();

        public bool Equals(ComputationNode<T>? x, ComputationNode<T>? y) => ReferenceEquals(x, y);

        public int GetHashCode(ComputationNode<T> obj) => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(obj);
    }
}
