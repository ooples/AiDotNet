using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;

namespace AiDotNet.Serialization;

/// <summary>
/// The string-keyed construction state a layer writes on save and a generated factory reads on load.
/// </summary>
/// <remarks>
/// <para>
/// <c>LayerStateGenerator</c> emits both halves of this contract. On save it emits a
/// <c>WriteConstructionState</c> override that calls <see cref="Format(object?)"/> /
/// <see cref="FormatType(object?)"/> for each <c>[LayerState]</c> member; on load it emits a factory
/// that calls <see cref="HasAll(string[])"/> and then the typed readers below. Both sides name the same
/// keys, so this type is the single place the wire encoding is decided.
/// </para>
/// <para>
/// EVERY VALUE IS A STRING, AND THE FORMAT IS INVARIANT. The bag is written into the same flat
/// <c>Dictionary&lt;string, string&gt;</c> that carries the rest of a model's metadata, so a model saved
/// under a comma-decimal locale must load under a period-decimal one. Every conversion here pins
/// <see cref="CultureInfo.InvariantCulture"/> for that reason, and floating-point values use the
/// round-trip format so a reload is bit-identical rather than merely close.
/// </para>
/// <para>
/// A MISSING OR MALFORMED KEY THROWS. The generated factory has already called
/// <see cref="HasAll(string[])"/> and returned <c>false</c> when a required key was absent, so reaching a
/// reader with bad state means the saved file disagrees with the compiled layer. Returning a default
/// there would rebuild a DIFFERENT layer than the one that was saved and report success, which is the
/// failure this type exists to make impossible.
/// </para>
/// <para><b>For Beginners:</b> When a layer is saved, its settings (how many neurons, which activation)
/// are written down as text. This class writes them down and reads them back.</para>
/// </remarks>
public sealed class LayerStateBag
{
    /// <summary>Separates the elements of an <c>int[]</c> value.</summary>
    private const char ArraySeparator = ',';

    private readonly IReadOnlyDictionary<string, string> _values;

    /// <summary>Wraps the metadata dictionary a layer was saved with.</summary>
    /// <param name="values">The saved key/value pairs. Not copied; treated as read-only.</param>
    public LayerStateBag(IReadOnlyDictionary<string, string> values)
    {
        _values = values ?? throw new ArgumentNullException(nameof(values));
    }

    /// <summary>True when every one of <paramref name="keys"/> is present.</summary>
    /// <remarks>
    /// The generated factory calls this before reading anything and returns <c>false</c> when it fails,
    /// so an older save that predates a new <c>[LayerState]</c> member falls back to the legacy
    /// reconstruction path instead of throwing.
    /// </remarks>
    public bool HasAll(params string[] keys)
    {
        if (keys is null) return false;
        foreach (var key in keys)
        {
            if (key is null || !_values.ContainsKey(key)) return false;
        }
        return true;
    }

    /// <summary>Reads a 32-bit integer.</summary>
    public int Int32(string key)
        => int.Parse(Require(key), NumberStyles.Integer, CultureInfo.InvariantCulture);

    /// <summary>Reads a 64-bit integer.</summary>
    public long Int64(string key)
        => long.Parse(Require(key), NumberStyles.Integer, CultureInfo.InvariantCulture);

    /// <summary>Reads a double-precision value.</summary>
    public double Double(string key)
        => double.Parse(Require(key), NumberStyles.Float, CultureInfo.InvariantCulture);

    /// <summary>Reads a single-precision value.</summary>
    public float Single(string key)
        => float.Parse(Require(key), NumberStyles.Float, CultureInfo.InvariantCulture);

    /// <summary>Reads a boolean.</summary>
    /// <remarks>
    /// Written by <see cref="Format(object?)"/> as <c>true</c>/<c>false</c> lowercase, so parsing is
    /// case-insensitive here rather than relying on <see cref="bool.Parse"/> alone matching a value some
    /// other writer produced.
    /// </remarks>
    public bool Boolean(string key)
    {
        var raw = Require(key);
        if (bool.TryParse(raw, out bool parsed)) return parsed;
        throw new FormatException($"Layer state key '{key}' holds '{raw}', which is not a boolean.");
    }

    /// <summary>Reads a string.</summary>
    public string String(string key) => Require(key);

    /// <summary>Reads a comma-separated <c>int[]</c>.</summary>
    /// <remarks>
    /// An empty string is an EMPTY array, not a one-element array containing a parse failure —
    /// <c>"".Split(',')</c> yields one empty entry, so the empty case is handled before splitting.
    /// </remarks>
    public int[] Int32Array(string key)
    {
        var raw = Require(key);
        if (raw.Length == 0) return [];

        var parts = raw.Split(ArraySeparator);
        var result = new int[parts.Length];
        for (int i = 0; i < parts.Length; i++)
        {
            result[i] = int.Parse(parts[i], NumberStyles.Integer, CultureInfo.InvariantCulture);
        }
        return result;
    }

    /// <summary>Reads an enum by member name.</summary>
    /// <remarks>
    /// By NAME, not by numeric value: a saved model must survive someone inserting a member into the
    /// middle of an enum, which renumbers every member after it.
    /// </remarks>
    public TEnum Enum<TEnum>(string key) where TEnum : struct, Enum
    {
        var raw = Require(key);
        if (System.Enum.TryParse<TEnum>(raw, ignoreCase: false, out var parsed)
            && System.Enum.IsDefined(typeof(TEnum), parsed))
        {
            return parsed;
        }
        throw new FormatException(
            $"Layer state key '{key}' holds '{raw}', which is not a member of {typeof(TEnum).Name}.");
    }

    /// <summary>Rebuilds a component (an activation function, a strategy object) from its saved type name.</summary>
    /// <remarks>
    /// <para>
    /// The saved value is a type name, and a type name from a file is UNTRUSTED INPUT. Instantiating
    /// whatever it happens to name is the classic deserialization gadget, so this constrains the
    /// resolved type twice: it must live under the <c>AiDotNet</c> namespace (the same rule
    /// <see cref="SafeSerializationBinder"/> applies), and it must be assignable to
    /// <typeparamref name="TComponent"/>, which is the compile-time type the generated factory is about
    /// to assign it to.
    /// </para>
    /// <para>
    /// Only a public parameterless constructor is used. Components that need arguments are not
    /// round-trippable this way, and <c>LayerStateGenerator</c> reports ADN0052 for them at build time
    /// rather than letting them fail here at load time.
    /// </para>
    /// </remarks>
    public TComponent Component<TComponent>(string key) where TComponent : class
    {
        var typeName = Require(key);

        var resolved = Type.GetType(typeName, throwOnError: false)
            ?? ResolveFromLoadedAssemblies(typeName);

        if (resolved is null)
        {
            throw new InvalidOperationException(
                $"Layer state key '{key}' names type '{typeName}', which could not be resolved.");
        }

        if (resolved.FullName is null || !resolved.FullName.StartsWith("AiDotNet.", StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                $"Layer state key '{key}' names type '{resolved.FullName ?? typeName}', which is outside the " +
                "AiDotNet namespace. Only AiDotNet components may be reconstructed from saved state.");
        }

        if (!typeof(TComponent).IsAssignableFrom(resolved))
        {
            throw new InvalidOperationException(
                $"Layer state key '{key}' names type '{resolved.FullName}', which is not a " +
                $"{typeof(TComponent).Name}.");
        }

        var instance = Activator.CreateInstance(resolved) as TComponent;
        return instance ?? throw new InvalidOperationException(
            $"Layer state key '{key}' names type '{resolved.FullName}', which has no usable " +
            "public parameterless constructor.");
    }

    /// <summary>Writes a value in the invariant round-trippable form the readers above expect.</summary>
    /// <remarks>
    /// <c>"R"</c> for <see cref="double"/> and <see cref="float"/>: the default <c>ToString()</c> rounds
    /// to 15 and 7 significant digits, so a saved learning rate or epsilon would come back CLOSE to the
    /// value that was saved rather than equal to it, and a reloaded model would drift from the one that
    /// was written.
    /// </remarks>
    public static string Format(object? value) => value switch
    {
        null => string.Empty,
        string s => s,
        bool b => b ? "true" : "false",
        double d => d.ToString("R", CultureInfo.InvariantCulture),
        float f => f.ToString("R", CultureInfo.InvariantCulture),
        int[] a => string.Join(ArraySeparator.ToString(), a.Select(x => x.ToString(CultureInfo.InvariantCulture))),
        Enum e => e.ToString(),
        IFormattable formattable => formattable.ToString(null, CultureInfo.InvariantCulture),
        _ => value.ToString() ?? string.Empty,
    };

    /// <summary>Writes a component as the type name <see cref="Component{TComponent}"/> reads back.</summary>
    /// <remarks>
    /// Full name plus the assembly's SIMPLE name, deliberately without version or public key: a model
    /// saved against one build of AiDotNet must load against the next, and an assembly-qualified name
    /// that pins a version would break on every release.
    /// </remarks>
    public static string FormatType(object? component)
    {
        if (component is null) return string.Empty;

        var type = component as Type ?? component.GetType();
        var assemblyName = type.GetTypeInfo().Assembly.GetName().Name;
        return string.IsNullOrEmpty(assemblyName)
            ? type.FullName ?? type.Name
            : $"{type.FullName}, {assemblyName}";
    }

    /// <summary>The value for <paramref name="key"/>, or a throw naming the key that was missing.</summary>
    private string Require(string key)
    {
        if (key is null) throw new ArgumentNullException(nameof(key));
        if (_values.TryGetValue(key, out var value) && value is not null) return value;

        throw new KeyNotFoundException(
            $"Layer state is missing required key '{key}'. The saved model does not match the compiled layer.");
    }

    /// <summary>Falls back to a scan when <see cref="Type.GetType(string, bool)"/> cannot see the assembly.</summary>
    /// <remarks>
    /// <see cref="Type.GetType(string, bool)"/> only probes the calling assembly and the core library, so
    /// a component defined in a sibling AiDotNet assembly resolves to null there. The scan is restricted
    /// to already-loaded assemblies; nothing here causes an assembly to be loaded from a name in a file.
    /// </remarks>
    private static Type? ResolveFromLoadedAssemblies(string typeName)
    {
        var comma = typeName.IndexOf(',');
        var bareName = comma < 0 ? typeName : typeName.Substring(0, comma).Trim();
        var assemblySimpleName = comma < 0 ? null : typeName.Substring(comma + 1).Trim();

        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            if (assemblySimpleName is not null
                && !string.Equals(assembly.GetName().Name, assemblySimpleName, StringComparison.Ordinal))
            {
                continue;
            }

            var found = assembly.GetType(bareName, throwOnError: false);
            if (found is not null) return found;
        }

        return null;
    }
}
