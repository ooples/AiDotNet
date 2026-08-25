using System.Globalization;
using System.Linq.Expressions;
using System.Reflection;
using System.Text;

namespace AiDotNet.Serialization;

/// <summary>
/// Saves and restores an expression tree a layer was constructed with.
/// </summary>
/// <remarks>
/// <para>
/// The middle of <see cref="DelegateState"/>'s three descriptions. A traced graph records what an
/// expression computed on one probe input; a method reference names a function that already exists.
/// An expression tree sits between them: the function as data, complete with its captured constants,
/// for a layer that was handed one before it was compiled.
/// </para>
/// <para>
/// It has to be captured at construction. Compiling an <see cref="Expression{TDelegate}"/> to a
/// delegate is one-way -- the tree is not recoverable from the result -- so a layer that wants this
/// takes the expression itself and keeps it.
/// </para>
/// <para>
/// Unlike a traced graph, an expression can name any method, so the allowlist here is not free the
/// way <c>TensorOperations</c> lookup is. Loading resolves methods only on types the host has
/// approved, and rejects the tree otherwise -- before <see cref="LambdaExpression.Compile()"/> is
/// ever called. Nothing is executed to decide this.
/// </para>
/// </remarks>
public static class ExpressionState
{
    /// <summary>
    /// Types whose methods a restored expression may call.
    /// </summary>
    /// <remarks>
    /// A saved model is data from somewhere else. Without this, a tree naming
    /// <c>File.Delete(string)</c> would resolve and run on the first forward pass -- the class of
    /// hazard that makes Keras's Lambda layer unsafe to load, arrived at from a different direction.
    /// The default is the assembly that defines the layers plus the framework's maths.
    /// </remarks>
    public static bool IsAllowed(Type type)
        => type.Assembly == typeof(ExpressionState).Assembly
           // The framework's maths now lives in its own package. Checking a single assembly's
           // identity stopped covering Tensor<T> when AiDotNet.Tensors was extracted, so a saved
           // LambdaLayer expression -- whose whole job is transforming tensors -- was rejected on
           // restore even though the layer had saved it happily. The boundary is unchanged in
           // spirit: first-party framework types, never arbitrary ones.
           || type.Assembly == typeof(AiDotNet.Tensors.LinearAlgebra.Tensor<double>).Assembly
           || type == typeof(Math)
           || type == typeof(MathF);

    /// <summary>Describes an expression tree, or returns empty when it uses something unsupported.</summary>
    /// <param name="expression">The expression the layer was constructed with.</param>
    /// <returns>The saved form, or <see cref="string.Empty"/>.</returns>
    /// <remarks>
    /// All-or-nothing, like the traced graph: a node this cannot record faithfully abandons the
    /// whole tree so the caller falls through to a weaker description, rather than saving a
    /// fragment that rebuilds into a different function.
    /// </remarks>
    public static string Save(LambdaExpression? expression)
    {
        if (expression is null || expression.Parameters.Count != 1) return string.Empty;

        var nodes = new List<string>();
        var ids = new Dictionary<Expression, int>(ReferenceComparer.Instance);
        var writer = new Writer(nodes, ids, expression.Parameters[0]);

        var root = writer.Visit(expression.Body);
        if (root < 0) return string.Empty;

        var sb = new StringBuilder();
        foreach (var node in nodes)
        {
            if (sb.Length > 0) sb.Append(';');
            sb.Append(node);
        }
        return sb.Append(';').Append(root).ToString();
    }

    /// <summary>Rebuilds the expression a saved tree records.</summary>
    /// <typeparam name="TDelegate">The delegate the expression is over.</typeparam>
    /// <param name="saved">The value written by <see cref="Save"/>.</param>
    /// <param name="layerName">The layer being rebuilt, named in any failure.</param>
    /// <param name="key">The constructor parameter, named in any failure.</param>
    /// <returns>The rebuilt expression.</returns>
    public static Expression<TDelegate> Load<TDelegate>(string? saved, string layerName, string key)
        where TDelegate : Delegate
    {
        if (string.IsNullOrEmpty(saved)) throw Bad(layerName, key, "nothing was recorded for it.");

        var invoke = typeof(TDelegate).GetMethod("Invoke")!;
        if (invoke.GetParameters().Length != 1)
            throw Bad(layerName, key, $"{typeof(TDelegate).Name} does not take exactly one argument.");

        var parameter = Expression.Parameter(invoke.GetParameters()[0].ParameterType, "x");
        var parts = saved!.Split(';');
        if (parts.Length < 2 || !int.TryParse(parts[parts.Length - 1], NumberStyles.Integer,
                CultureInfo.InvariantCulture, out var rootId))
            throw Bad(layerName, key, "its recorded tree has no root.");

        var built = new Dictionary<int, Expression>();
        foreach (var step in parts.Take(parts.Length - 1))
        {
            if (step.Length == 0 || step == ">") continue;

            var eq = step.IndexOf('=');
            if (eq < 0 || !int.TryParse(step.Substring(0, eq), out var id))
                throw Bad(layerName, key, $"a recorded node '{step}' is malformed.");

            built[id] = Rebuild(step.Substring(eq + 1), built, parameter, layerName, key);
        }

        if (!built.TryGetValue(rootId, out var body))
            throw Bad(layerName, key, "its recorded root node was never produced.");

        return Expression.Lambda<TDelegate>(body, parameter);
    }

    private static Expression Rebuild(
        string body, Dictionary<int, Expression> built, ParameterExpression parameter,
        string layerName, string key)
    {
        var fields = body.Split('|');
        switch (fields[0])
        {
            case "P":
                return parameter;

            case "C":
            {
                var type = Resolve(fields[1], layerName, key);
                return Expression.Constant(ParseConstant(fields[2], type, layerName, key), type);
            }

            case "B":
            {
                var op = (ExpressionType)Enum.Parse(typeof(ExpressionType), fields[1]);
                return Expression.MakeBinary(op, built[int.Parse(fields[2])], built[int.Parse(fields[3])]);
            }

            case "U":
            {
                var op = (ExpressionType)Enum.Parse(typeof(ExpressionType), fields[1]);
                return Expression.MakeUnary(op, built[int.Parse(fields[3])], Resolve(fields[2], layerName, key));
            }

            case "M":
            {
                var declaring = Resolve(fields[1], layerName, key);

                // Checked before the method is bound and long before Compile() -- nothing from the
                // saved tree is executed to reach this decision.
                if (!IsAllowed(declaring))
                    throw Bad(layerName, key,
                        $"its recorded tree calls into '{declaring.FullName}', which is not a type a "
                        + "restored expression is allowed to call.");

                var parameterTypes = fields[3].Length == 0
                    ? Array.Empty<Type>()
                    : fields[3].Split('~').Select(n => Resolve(n, layerName, key)).ToArray();

                var method = declaring.GetMethod(
                    fields[2],
                    BindingFlags.Public | BindingFlags.Static | BindingFlags.Instance,
                    binder: null, types: parameterTypes, modifiers: null)
                    ?? throw Bad(layerName, key,
                        $"'{declaring.Name}' no longer declares '{fields[2]}' with the saved signature.");

                var instanceId = int.Parse(fields[4], CultureInfo.InvariantCulture);
                var arguments = fields[5].Length == 0
                    ? Array.Empty<Expression>()
                    : fields[5].Split(',').Select(s => built[int.Parse(s, CultureInfo.InvariantCulture)]).ToArray();

                return instanceId < 0
                    ? Expression.Call(method, arguments)
                    : Expression.Call(built[instanceId], method, arguments);
            }

            default:
                throw Bad(layerName, key, $"a recorded node '{body}' is of a kind this version does not know.");
        }
    }

    private static Type Resolve(string name, string layerName, string key)
        => Type.GetType(name, throwOnError: false)
           ?? throw Bad(layerName, key, $"the type '{name}' could not be loaded.");

    private static object? ParseConstant(string raw, Type type, string layerName, string key)
    {
        if (raw == "~") return null;
        if (type == typeof(string)) return raw;
        if (type.IsEnum) return Enum.Parse(type, raw, ignoreCase: true);

        try
        {
            return Convert.ChangeType(raw, Nullable.GetUnderlyingType(type) ?? type, CultureInfo.InvariantCulture);
        }
        catch (Exception)
        {
            throw Bad(layerName, key, $"the constant '{raw}' is not a {type.Name}.");
        }
    }

    private static InvalidOperationException Bad(string layerName, string key, string why)
        => new($"Cannot rebuild {layerName}: constructor parameter '{key}' was saved as an expression, but {why}");

    /// <summary>Flattens a tree into numbered nodes, refusing anything that will not round-trip.</summary>
    private sealed class Writer
    {
        private readonly List<string> _nodes;
        private readonly Dictionary<Expression, int> _ids;
        private readonly ParameterExpression _parameter;

        internal Writer(List<string> nodes, Dictionary<Expression, int> ids, ParameterExpression parameter)
        {
            _nodes = nodes;
            _ids = ids;
            _parameter = parameter;
        }

        /// <returns>The node's id, or -1 when the expression cannot be recorded.</returns>
        /// <summary>
        /// Whether the subtree never reaches the lambda's parameter, and so has a value already.
        /// </summary>
        private bool IsClosed(Expression node) => node switch
        {
            ParameterExpression => false,
            ConstantExpression => true,
            MemberExpression m => m.Expression is null || IsClosed(m.Expression),
            UnaryExpression u => IsClosed(u.Operand),
            BinaryExpression b => IsClosed(b.Left) && IsClosed(b.Right),
            MethodCallExpression c => (c.Object is null || IsClosed(c.Object)) && c.Arguments.All(IsClosed),
            _ => false,
        };

        internal int Visit(Expression node)
        {
            if (_ids.TryGetValue(node, out var existing)) return existing;

            // A captured local is not a constant node: the compiler lifts it to a field on a
            // closure class, so `x * scale` reads as a member access. Any subtree that never
            // reaches the parameter already has a value, so it is evaluated here and recorded as
            // the constant it is -- which is what lets a closure round-trip at all, and the reason
            // this tier exists rather than only naming methods.
            if (!ReferenceEquals(node, _parameter) && node is not ConstantExpression && IsClosed(node))
            {
                object? value;
                try
                {
                    value = Expression.Lambda(node).Compile().DynamicInvoke();
                }
                catch (Exception)
                {
                    return -1;
                }

                return Visit(Expression.Constant(value, node.Type));
            }

            string? encoded = node switch
            {
                ParameterExpression p when ReferenceEquals(p, _parameter) => "P",
                ConstantExpression c => Constant(c),
                BinaryExpression b when b.Method is null && b.Conversion is null => Binary(b),
                UnaryExpression u => Unary(u),
                MethodCallExpression m => Call(m),
                _ => null,
            };

            if (encoded is null) return -1;

            var id = _ids.Count;
            _ids[node] = id;
            _nodes.Add(id.ToString(CultureInfo.InvariantCulture) + "=" + encoded);
            return id;
        }

        private string? Constant(ConstantExpression c)
        {
            var name = c.Type.AssemblyQualifiedName;
            if (name is null) return null;

            // A captured object would have to be serialized whole, which is what the traced graph
            // declines to do for the same reason: it is not construction state.
            if (c.Value is not null && !c.Type.IsPrimitive && c.Type != typeof(string) && !c.Type.IsEnum)
                return null;

            var value = c.Value switch
            {
                null => "~",
                double d => d.ToString("R", CultureInfo.InvariantCulture),
                float f => f.ToString("R", CultureInfo.InvariantCulture),
                IFormattable v => v.ToString(null, CultureInfo.InvariantCulture),
                var v => v.ToString(),
            };

            return value is null || value.IndexOfAny(['|', ';', '=']) >= 0 ? null : $"C|{name}|{value}";
        }

        private string? Binary(BinaryExpression b)
        {
            var left = Visit(b.Left);
            var right = Visit(b.Right);
            return left < 0 || right < 0 ? null : $"B|{b.NodeType}|{left}|{right}";
        }

        private string? Unary(UnaryExpression u)
        {
            var name = u.Type.AssemblyQualifiedName;
            if (name is null || u.Method is not null) return null;

            var operand = Visit(u.Operand);
            return operand < 0 ? null : $"U|{u.NodeType}|{name}|{operand}";
        }

        private string? Call(MethodCallExpression m)
        {
            var declaring = m.Method.DeclaringType?.AssemblyQualifiedName;
            if (declaring is null || m.Method.IsGenericMethod) return null;

            var instance = -1;
            if (m.Object is not null)
            {
                instance = Visit(m.Object);
                if (instance < 0) return null;
            }

            var arguments = new List<int>();
            foreach (var argument in m.Arguments)
            {
                var id = Visit(argument);
                if (id < 0) return null;
                arguments.Add(id);
            }

            var parameters = string.Join("~", m.Method.GetParameters()
                .Select(p => p.ParameterType.AssemblyQualifiedName ?? string.Empty));

            return $"M|{declaring}|{m.Method.Name}|{parameters}|{instance}|{string.Join(",", arguments)}";
        }
    }

    private sealed class ReferenceComparer : IEqualityComparer<Expression>
    {
        internal static readonly ReferenceComparer Instance = new();

        public bool Equals(Expression? x, Expression? y) => ReferenceEquals(x, y);

        public int GetHashCode(Expression obj) => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(obj);
    }
}
