using System.Reflection;
using System.Runtime.CompilerServices;

namespace AiDotNet.Serialization;

/// <summary>
/// Saves and restores a delegate a layer was constructed with.
/// </summary>
/// <remarks>
/// <para>
/// A delegate is the one kind of construction state that cannot simply be written down, and the
/// established answers are both unsatisfying. Python's <c>pickle</c> refuses a lambda outright and
/// stores a module-level function as a name reference. Keras goes further and marshals the Lambda
/// layer's Python bytecode into the model file, which is why loading one is arbitrary code
/// execution and why <c>safe_mode=True</c> blocks it by default -- it was still bypassable
/// (CVE-2025-9906). PyTorch avoids the question: <c>copy.deepcopy</c> treats a function as atomic
/// and returns the same object, so a clone aliases the delegate and a save never round-trips it.
/// </para>
/// <para>
/// .NET makes better available, because a delegate is not opaque here: it is a
/// <see cref="MethodInfo"/> plus a target. So this tries progressively weaker descriptions and
/// keeps the first that fits, rather than marshalling code:
/// </para>
/// <list type="number">
/// <item>the traced computation graph, when the layer was given a traceable expression;</item>
/// <item>the expression tree, when the layer was given one rather than a compiled delegate;</item>
/// <item>a method reference, for a named static method -- pickle's answer, without the pickle.</item>
/// </list>
/// <para>
/// When none fits, the delegate is reported as unsaveable rather than written as something that
/// will not come back. Cloning a layer in memory does not go through here at all: it hands the
/// live delegate over, which is what PyTorch's deepcopy does and is always correct in-process.
/// </para>
/// </remarks>
public static class DelegateState
{
    /// <summary>Marks a saved value as a reference to a named method.</summary>
    public const string MethodScheme = "method:";

    /// <summary>Marks a saved value as a serialized expression tree.</summary>
    public const string ExpressionScheme = "expr:";

    /// <summary>Marks a saved value as a traced computation graph.</summary>
    public const string GraphScheme = "graph:";

    /// <summary>
    /// Describes <paramref name="value"/> well enough to rebuild it, or returns empty when no
    /// description fits.
    /// </summary>
    /// <param name="value">The delegate the layer was constructed with.</param>
    /// <returns>The saved form, or <see cref="string.Empty"/> when it cannot be saved.</returns>
    public static string Save(Delegate? value)
    {
        if (value is null) return string.Empty;

        // Tiers 1 and 2 are chosen at construction time, not discovered here: an expression tree
        // and a traceable graph are both lost the moment they are compiled to a Func, so a layer
        // that has one records it directly. This is the fallback that works on any delegate.
        return SaveMethodReference(value) ?? string.Empty;
    }

    /// <summary>
    /// A named static method, as its declaring type, name and parameter types.
    /// </summary>
    /// <param name="value">The delegate to describe.</param>
    /// <returns>The reference, or <c>null</c> when this delegate is not a named static method.</returns>
    /// <remarks>
    /// The parameter types are recorded because a name alone is ambiguous across overloads, and
    /// picking the wrong overload at load time would rebuild a layer that computes something else.
    /// </remarks>
    private static string? SaveMethodReference(Delegate value)
    {
        var method = value.Method;
        var declaring = method.DeclaringType;
        if (declaring is null) return null;

        // A lambda body lives on a compiler-generated closure class, under a name that is not
        // stable across a recompile. Naming it would produce a reference that resolves today and
        // silently fails to resolve after an unrelated edit.
        if (declaring.IsDefined(typeof(CompilerGeneratedAttribute), inherit: false)
            || declaring.Name.IndexOf('<') >= 0
            || method.Name.IndexOf('<') >= 0)
            return null;

        // An instance method would need its receiver rebuilt too, which is the whole problem again.
        if (!method.IsStatic || value.Target is not null) return null;

        var owner = declaring.AssemblyQualifiedName;
        if (string.IsNullOrEmpty(owner)) return null;

        var parameters = string.Join(";", method.GetParameters()
            .Select(p => p.ParameterType.AssemblyQualifiedName ?? p.ParameterType.FullName ?? string.Empty));

        return MethodScheme + owner + "|" + method.Name + "|" + parameters;
    }

    /// <summary>Rebuilds a delegate from its saved form.</summary>
    /// <typeparam name="TDelegate">The delegate type the constructor takes.</typeparam>
    /// <param name="saved">The value written by <see cref="Save"/>.</param>
    /// <param name="layerName">The layer being rebuilt, named in any failure.</param>
    /// <param name="key">The constructor parameter, named in any failure.</param>
    /// <returns>The rebuilt delegate.</returns>
    /// <exception cref="InvalidOperationException">The delegate could not be rebuilt.</exception>
    public static TDelegate Load<TDelegate>(string? saved, string layerName, string key)
        where TDelegate : Delegate
    {
        if (string.IsNullOrEmpty(saved))
            throw Unsaveable(layerName, key,
                "nothing was recorded for it -- the layer was built with a lambda or a closure, "
                + "which has no name to refer to. Pass a named static method, an expression tree, "
                + "or a traceable expression if this layer needs to survive a save.");

        if (saved!.StartsWith(MethodScheme, StringComparison.Ordinal))
            return LoadMethodReference<TDelegate>(saved.Substring(MethodScheme.Length), layerName, key);

        throw Unsaveable(layerName, key, $"its saved form '{Excerpt(saved)}' is not a form this version understands.");
    }

    private static TDelegate LoadMethodReference<TDelegate>(string reference, string layerName, string key)
        where TDelegate : Delegate
    {
        var parts = reference.Split('|');
        if (parts.Length != 3)
            throw Unsaveable(layerName, key, $"its saved method reference '{Excerpt(reference)}' is malformed.");

        var declaring = Type.GetType(parts[0], throwOnError: false);
        if (declaring is null)
            throw Unsaveable(layerName, key, $"the type '{Excerpt(parts[0])}' that declared it could not be loaded.");

        var parameterTypes = parts[2].Length == 0
            ? []
            : parts[2].Split(';').Select(n => Type.GetType(n, throwOnError: false)).ToArray();

        if (parameterTypes.Any(t => t is null))
            throw Unsaveable(layerName, key, "one of its parameter types could not be loaded.");

        var method = declaring.GetMethod(
            parts[1],
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static,
            binder: null,
            types: parameterTypes!,
            modifiers: null);

        if (method is null)
            throw Unsaveable(layerName, key,
                $"'{declaring.Name}' no longer declares a static method '{parts[1]}' with the saved signature.");

        try
        {
            return (TDelegate)Delegate.CreateDelegate(typeof(TDelegate), method);
        }
        catch (ArgumentException ex)
        {
            throw Unsaveable(layerName, key,
                $"'{declaring.Name}.{parts[1]}' no longer matches {typeof(TDelegate).Name}: {ex.Message}");
        }
    }

    private static InvalidOperationException Unsaveable(string layerName, string key, string why)
        => new($"Cannot rebuild {layerName}: constructor parameter '{key}' is a delegate and {why}");

    private static string Excerpt(string value) => value.Length <= 120 ? value : value.Substring(0, 117) + "...";
}
