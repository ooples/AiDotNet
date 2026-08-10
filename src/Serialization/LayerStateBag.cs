using System.Globalization;

namespace AiDotNet.Serialization;

/// <summary>
/// The saved constructor state of a single layer, as read back at deserialization time.
/// </summary>
/// <remarks>
/// <para>
/// Generated factories built from <see cref="AiDotNet.Attributes.LayerStateAttribute"/> read their
/// constructor arguments through this type. Every accessor throws a message naming the layer, the
/// key and the keys that <i>were</i> present when a value is missing or unparseable, because the
/// alternative — silently substituting a default — is how a restored layer ends up quietly the wrong
/// size and only fails much later in a numeric comparison.
/// </para>
/// <para>
/// Lookup is case-insensitive so that generated readers keyed on a camelCase parameter name find the
/// PascalCase keys existing hand-written <c>GetMetadata</c> overrides already write.
/// </para>
/// </remarks>
public readonly struct LayerStateBag
{
    private readonly Dictionary<string, object>? _values;
    private readonly string _layerName;

    /// <summary>Creates a bag over the metadata restored for a layer.</summary>
    /// <param name="values">The saved metadata, or <c>null</c> when none was stored.</param>
    /// <param name="layerName">The layer type name, used only to make failures legible.</param>
    public LayerStateBag(Dictionary<string, object>? values, string layerName)
    {
        if (values is null)
        {
            _values = null;
        }
        else
        {
            // Copied entry by entry rather than through the copy constructor: a layer that also
            // hand-writes GetMetadata records the same value under its own casing ("DropoutRate"
            // beside the generated "dropoutRate"), and the copy constructor treats those as a
            // duplicate key and throws. Both hold the same value, so last one wins.
            _values = new Dictionary<string, object>(values.Count, StringComparer.OrdinalIgnoreCase);
            foreach (var kvp in values)
            {
                _values[kvp.Key] = kvp.Value;
            }
        }

        _layerName = layerName;
    }

    /// <summary>The layer type name this state belongs to.</summary>
    public string LayerName => _layerName;

    /// <summary><c>true</c> when every one of <paramref name="keys"/> is present.</summary>
    /// <param name="keys">The metadata keys a rebuild requires.</param>
    /// <returns><c>true</c> when all are present.</returns>
    /// <remarks>
    /// Guards against rebuilding a layer from state that belongs to something else. A wrapper such
    /// as <c>BidirectionalLayer</c> reconstructs its inner layer recursively and passes ITS OWN
    /// metadata down, so a factory keyed on the inner layer's parameter names can be handed a
    /// dictionary that never described it. Reporting "not mine" lets the caller fall back to the
    /// shape-derived path that has always handled the recursive case.
    /// </remarks>
    public bool HasAll(params string[] keys)
    {
        foreach (var key in keys)
        {
            if (!Has(key)) return false;
        }
        return true;
    }

    /// <summary><c>true</c> when a value is present for <paramref name="key"/>.</summary>
    public bool Has(string key) => TryRaw(key, out _);

    /// <summary>The key under which a nested layer's concrete type is recorded.</summary>
    /// <remarks>
    /// <see cref="AiDotNet.NeuralNetworks.Layers.LayerBase{T}"/>'s own metadata records the types of
    /// its activations but not of the layer itself, because until now nothing rebuilt a layer from
    /// inside another layer's state.
    /// </remarks>
    public const string TypeKey = "$type";

    /// <summary>
    /// The state of one nested value, as its own bag with the prefix removed.
    /// </summary>
    /// <param name="key">The parameter the nested state was written under.</param>
    /// <returns>A bag over just that parameter's keys.</returns>
    /// <remarks>
    /// A composite layer records a child under a <c>child.</c> key namespace, so the child's factory
    /// reads its own parameter names unchanged. Previously a wrapper passed its OWN metadata down,
    /// so a child factory could be handed a dictionary that never described it -- which is what
    /// <see cref="HasAll"/> exists to detect and what this removes the need for.
    /// </remarks>
    public LayerStateBag Nested(string key)
    {
        var prefix = key + ".";
        var sub = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);
        if (_values is not null)
        {
            foreach (var kvp in _values)
            {
                if (kvp.Key.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                    sub[kvp.Key.Substring(prefix.Length)] = kvp.Value;
            }
        }

        return new LayerStateBag(sub, _layerName + "." + key);
    }

    /// <summary>The concrete type recorded for a nested layer, if one was saved and can be loaded.</summary>
    /// <param name="key">The parameter the nested layer was written under.</param>
    /// <returns>The child's type, or <c>null</c>.</returns>
    public Type? NestedType(string key)
    {
        if (!TryRaw(key + "." + TypeKey, out var v)) return null;
        var name = AsText(v);
        return name.Length == 0 ? null : Type.GetType(name, throwOnError: false);
    }

    /// <summary>How many elements were saved for a nested collection.</summary>
    /// <param name="key">The parameter the collection was written under.</param>
    /// <returns>The element count, or -1 when nothing was saved.</returns>

    /// <summary>Records a child layer's own construction state under a nested key namespace.</summary>
    /// <typeparam name="TNum">The child's numeric type.</typeparam>
    /// <param name="metadata">The parent's metadata, written in place.</param>
    /// <param name="key">The constructor parameter the child was passed as.</param>
    /// <param name="child">The child layer, or <c>null</c>.</param>
    /// <remarks>
    /// The child's concrete type goes in explicitly because a layer's own metadata does not record
    /// it -- it records its activations' types, which was enough while nothing rebuilt a layer from
    /// inside another's state. Everything else is the child's own <c>GetMetadata</c>, so a composite
    /// layer stays rebuildable exactly as far as its children are, with no separate mechanism to
    /// keep in step.
    /// </remarks>
    public static void WriteNested<TNum>(
        Dictionary<string, string> metadata,
        string key,
        AiDotNet.NeuralNetworks.Layers.LayerBase<TNum>? child)
    {
        if (child is null) return;

        var type = child.GetType();
        metadata[key + "." + TypeKey] = type.AssemblyQualifiedName ?? type.FullName ?? string.Empty;
        foreach (var kvp in child.GetMetadata())
        {
            metadata[key + "." + kvp.Key] = kvp.Value;
        }
    }

    /// <summary>Records a sequence of child layers, plus the count a rebuild reads back.</summary>
    /// <typeparam name="TNum">The children's numeric type.</typeparam>
    /// <param name="metadata">The parent's metadata, written in place.</param>
    /// <param name="key">The constructor parameter the children were passed as.</param>
    /// <param name="children">The child layers, or <c>null</c>.</param>
    /// <remarks>
    /// The count is written even when zero: how many children the layer was built with is itself
    /// construction state, and a mixture-of-experts rebuilt with a different expert count is wrong
    /// in a way no later shape check would catch.
    /// </remarks>
    public static void WriteNestedRange<TNum>(
        Dictionary<string, string> metadata,
        string key,
        System.Collections.Generic.IEnumerable<object?>? children)
    {
        if (children is null) return;

        var index = 0;
        foreach (var child in children)
        {
            WriteNested(metadata, key + "." + index.ToString(CultureInfo.InvariantCulture),
                child as AiDotNet.NeuralNetworks.Layers.LayerBase<TNum>);
            index++;
        }

        metadata[key + ".count"] = index.ToString(CultureInfo.InvariantCulture);
    }

    public int NestedCount(string key) => Has(key + ".count") ? Int32(key + ".count") : -1;


    private bool TryRaw(string key, out object value)
    {
        value = null!;
        if (_values is null || !_values.TryGetValue(key, out var v) || v is null) return false;
        value = v;
        return true;
    }

    private string Describe()
        => _values is null || _values.Count == 0
            ? "no metadata was saved for this layer at all"
            : "saved keys were: " + string.Join(", ", _values.Keys.OrderBy(k => k, StringComparer.Ordinal));

    private InvalidOperationException Missing(string key, string wanted)
        => new(
            $"Cannot rebuild {_layerName}: its constructor needs '{key}' ({wanted}), but {Describe()}. " +
            "This value is written at save time by the generated GetMetadata override for parameters " +
            "marked [LayerState]. A payload saved before that parameter was marked will not contain it.");

    /// <summary>Reads a required 32-bit integer.</summary>
    /// <param name="key">The metadata key.</param>
    /// <returns>The stored value.</returns>
    public int Int32(string key)
    {
        if (!TryRaw(key, out var v)) throw Missing(key, "an integer");
        if (v is int i) return i;
        if (v is long l && l >= int.MinValue && l <= int.MaxValue) return (int)l;
        if (int.TryParse(AsText(v), NumberStyles.Integer, CultureInfo.InvariantCulture, out var p)) return p;
        throw Unparseable(key, v, "an integer");
    }

    /// <summary>Reads a 32-bit integer, or <paramref name="fallback"/> when it was not saved.</summary>
    /// <param name="key">The metadata key.</param>
    /// <param name="fallback">Value to use when the key is absent.</param>
    /// <returns>The stored value, or the fallback.</returns>
    public int Int32(string key, int fallback) => Has(key) ? Int32(key) : fallback;

    /// <summary>Reads a required 64-bit integer.</summary>
    /// <param name="key">The metadata key.</param>
    /// <returns>The stored value.</returns>
    public long Int64(string key)
    {
        if (!TryRaw(key, out var v)) throw Missing(key, "an integer");
        if (v is long l) return l;
        if (v is int i) return i;
        if (long.TryParse(AsText(v), NumberStyles.Integer, CultureInfo.InvariantCulture, out var p)) return p;
        throw Unparseable(key, v, "an integer");
    }

    /// <summary>Reads a required double.</summary>
    /// <param name="key">The metadata key.</param>
    /// <returns>The stored value.</returns>
    public double Double(string key)
    {
        if (!TryRaw(key, out var v)) throw Missing(key, "a number");
        if (v is double d) return d;
        if (v is float f) return f;
        if (double.TryParse(AsText(v), NumberStyles.Float, CultureInfo.InvariantCulture, out var p)) return p;
        throw Unparseable(key, v, "a number");
    }

    /// <summary>Reads a double, or <paramref name="fallback"/> when it was not saved.</summary>
    /// <param name="key">The metadata key.</param>
    /// <param name="fallback">Value to use when the key is absent.</param>
    /// <returns>The stored value, or the fallback.</returns>
    public double Double(string key, double fallback) => Has(key) ? Double(key) : fallback;

    /// <summary>Reads a required single-precision float.</summary>
    /// <param name="key">The metadata key.</param>
    /// <returns>The stored value.</returns>
    public float Single(string key) => (float)Double(key);

    /// <summary>Reads a required boolean.</summary>
    /// <param name="key">The metadata key.</param>
    /// <returns>The stored value.</returns>
    public bool Boolean(string key)
    {
        if (!TryRaw(key, out var v)) throw Missing(key, "true or false");
        if (v is bool b) return b;
        if (bool.TryParse(AsText(v), out var p)) return p;
        throw Unparseable(key, v, "true or false");
    }

    /// <summary>Reads a boolean, or <paramref name="fallback"/> when it was not saved.</summary>
    /// <param name="key">The metadata key.</param>
    /// <param name="fallback">Value to use when the key is absent.</param>
    /// <returns>The stored value, or the fallback.</returns>
    public bool Boolean(string key, bool fallback) => Has(key) ? Boolean(key) : fallback;

    /// <summary>Reads a required string.</summary>
    /// <param name="key">The metadata key.</param>
    /// <returns>The stored value.</returns>
    public string String(string key)
        => TryRaw(key, out var v) ? AsText(v) : throw Missing(key, "text");

    /// <summary>Reads a string, or <paramref name="fallback"/> when it was not saved.</summary>
    /// <param name="key">The metadata key.</param>
    /// <param name="fallback">Value to use when the key is absent.</param>
    /// <returns>The stored value, or the fallback.</returns>
    public string? String(string key, string? fallback) => Has(key) ? String(key) : fallback;

    /// <summary>Reads a required enum value.</summary>
    /// <typeparam name="TEnum">The enum type.</typeparam>
    /// <param name="key">The metadata key.</param>
    /// <returns>The stored value.</returns>
    public TEnum Enum<TEnum>(string key) where TEnum : struct, Enum
    {
        if (!TryRaw(key, out var v)) throw Missing(key, $"one of {string.Join("/", System.Enum.GetNames(typeof(TEnum)))}");
        if (v is TEnum e) return e;
        if (System.Enum.TryParse<TEnum>(AsText(v), ignoreCase: true, out var p)) return p;
        throw Unparseable(key, v, $"one of {string.Join("/", System.Enum.GetNames(typeof(TEnum)))}");
    }

    /// <summary>Reads an enum value, or <paramref name="fallback"/> when it was not saved.</summary>
    /// <typeparam name="TEnum">The enum type.</typeparam>
    /// <param name="key">The metadata key.</param>
    /// <param name="fallback">Value to use when the key is absent.</param>
    /// <returns>The stored value, or the fallback.</returns>
    public TEnum Enum<TEnum>(string key, TEnum fallback) where TEnum : struct, Enum
        => Has(key) ? Enum<TEnum>(key) : fallback;

    /// <summary>Reads a required integer array, stored comma-separated.</summary>
    /// <param name="key">The metadata key.</param>
    /// <returns>The stored value.</returns>
    public int[] Int32Array(string key)
    {
        if (!TryRaw(key, out var v)) throw Missing(key, "a comma-separated list of integers");
        if (v is int[] arr) return arr;

        var text = AsText(v);
        if (text.Length == 0) return [];

        var parts = text.Split([',', ' '], StringSplitOptions.RemoveEmptyEntries);
        var result = new int[parts.Length];
        for (int i = 0; i < parts.Length; i++)
        {
            if (!int.TryParse(parts[i], NumberStyles.Integer, CultureInfo.InvariantCulture, out result[i]))
                throw Unparseable(key, v, "a comma-separated list of integers");
        }
        return result;
    }

    /// <summary>Reads a required array of booleans, such as a per-block flag.</summary>
    /// <param name="key">The metadata key.</param>
    /// <returns>The stored values.</returns>
    public bool[] BooleanArray(string key)
    {
        const string wanted = "a comma-separated list of true/false";
        if (!TryRaw(key, out var v)) throw Missing(key, wanted);
        if (v is bool[] arr) return arr;

        var text = AsText(v);
        if (text.Length == 0) return [];

        var parts = text.Split([',', ' '], StringSplitOptions.RemoveEmptyEntries);
        var result = new bool[parts.Length];
        for (int i = 0; i < parts.Length; i++)
        {
            if (!bool.TryParse(parts[i], out result[i])) throw Unparseable(key, v, wanted);
        }
        return result;
    }

    /// <summary>Reads a required array of doubles.</summary>
    /// <param name="key">The metadata key.</param>
    /// <returns>The stored values.</returns>
    public double[] DoubleArray(string key)
    {
        const string wanted = "a comma-separated list of numbers";
        if (!TryRaw(key, out var v)) throw Missing(key, wanted);
        if (v is double[] arr) return arr;

        var text = AsText(v);
        if (text.Length == 0) return [];

        var parts = text.Split([',', ' '], StringSplitOptions.RemoveEmptyEntries);
        var result = new double[parts.Length];
        for (int i = 0; i < parts.Length; i++)
        {
            if (!double.TryParse(parts[i], NumberStyles.Float, CultureInfo.InvariantCulture, out result[i]))
                throw Unparseable(key, v, wanted);
        }
        return result;
    }

    /// <summary>Reads a required jagged array of integers, such as one shape per input.</summary>
    /// <param name="key">The metadata key.</param>
    /// <returns>The stored rows.</returns>
    /// <remarks>
    /// Rows are separated by ';' and values within a row by ','. The row count is meaningful on its
    /// own -- a merge layer's input count is its outer length -- so an empty row is preserved as an
    /// empty row rather than dropped.
    /// </remarks>
    public int[][] Int32Jagged(string key)
    {
        const string wanted = "semicolon-separated rows of comma-separated integers";
        if (!TryRaw(key, out var v)) throw Missing(key, wanted);
        if (v is int[][] jagged) return jagged;

        var text = AsText(v);
        if (text.Length == 0) return [];

        var rows = text.Split(';');
        var result = new int[rows.Length][];
        for (int r = 0; r < rows.Length; r++)
        {
            var parts = rows[r].Split([',', ' '], StringSplitOptions.RemoveEmptyEntries);
            result[r] = new int[parts.Length];
            for (int i = 0; i < parts.Length; i++)
            {
                if (!int.TryParse(parts[i], NumberStyles.Integer, CultureInfo.InvariantCulture, out result[r][i]))
                    throw Unparseable(key, v, wanted);
            }
        }
        return result;
    }


    /// <summary>Reads an integer array, or <paramref name="fallback"/> when it was not saved.</summary>
    /// <param name="key">The metadata key.</param>
    /// <param name="fallback">Value to use when the key is absent.</param>
    /// <returns>The stored value, or the fallback.</returns>
    public int[]? Int32Array(string key, int[]? fallback) => Has(key) ? Int32Array(key) : fallback;

    /// <summary>
    /// Rebuilds a pluggable component (an RBF kernel, a distance metric, ...) from the concrete
    /// type recorded at save time.
    /// </summary>
    /// <typeparam name="TComponent">The interface the constructor takes.</typeparam>
    /// <param name="key">The metadata key.</param>
    /// <returns>
    /// The saved implementation, or <c>null</c> when none was recorded so the constructor's own
    /// default applies.
    /// </returns>
    /// <remarks>
    /// Recording the concrete type is what separates "reconstructable" from "reconstructed
    /// correctly": a layer built with a Multiquadric kernel that reloads as the default Gaussian is
    /// a different function, and nothing downstream would report it. This mirrors how the base
    /// layer already round-trips activations.
    /// </remarks>
    public TComponent? Component<TComponent>(string key) where TComponent : class
    {
        if (!TryRaw(key, out var v)) return null;

        var typeName = AsText(v);
        if (typeName.Length == 0) return null;

        var type = Type.GetType(typeName);
        if (type is null)
        {
            throw new InvalidOperationException(
                $"Cannot rebuild {_layerName}: it was saved using '{typeName}' for '{key}', but that " +
                "type could not be loaded. Substituting a default would silently change what the " +
                "layer computes, so the rebuild fails instead.");
        }

        if (type.IsGenericTypeDefinition)
        {
            var args = typeof(TComponent).GetGenericArguments();
            if (args.Length > 0) type = type.MakeGenericType(args);
        }

        return Activator.CreateInstance(type) as TComponent;
    }
    /// <summary>Reads a required component, failing legibly rather than returning null.</summary>
    /// <typeparam name="TComponent">The interface the constructor takes.</typeparam>
    /// <param name="key">The metadata key.</param>
    /// <returns>The rebuilt component.</returns>
    /// <remarks>
    /// <see cref="Component{TComponent}"/> returns <c>null</c> both when nothing was saved and when
    /// the saved type cannot be loaded, which is right for a parameter that accepts null. Handing
    /// that null to a constructor that does not is how a layer ends up half-built and fails much
    /// later somewhere unrelated, so a non-nullable parameter reads through here instead.
    /// </remarks>
    public TComponent ComponentRequired<TComponent>(string key) where TComponent : class
        => Component<TComponent>(key)
           ?? throw new InvalidOperationException(
               $"Cannot rebuild {_layerName}: metadata key '{key}' should name a " +
               $"{typeof(TComponent).Name} implementation, but " +
               (Has(key)
                   ? $"the saved type '{AsText(RawOrEmpty(key))}' could not be loaded."
                   : $"no value was saved for it -- {Describe()}."));

    private object RawOrEmpty(string key) => TryRaw(key, out var v) ? v : string.Empty;
    /// <summary>Casts a restored activation to the interface a constructor requires.</summary>
    /// <typeparam name="TActivation">The activation interface the parameter takes.</typeparam>
    /// <param name="value">The activation restored alongside the layer, if any.</param>
    /// <param name="parameterName">The constructor parameter, named in the failure.</param>
    /// <param name="layerName">The layer being rebuilt, named in the failure.</param>
    /// <returns>The activation, typed as the constructor wants it.</returns>
    /// <remarks>
    /// The restored activation arrives as <c>object?</c>, so <c>as</c> yields null both when none
    /// was saved and when a scalar activation was saved for a vector parameter. A constructor that
    /// does not accept null gets told which of those happened instead of a null argument.
    /// </remarks>
    public static TActivation RequireActivation<TActivation>(object? value, string parameterName, string layerName)
        where TActivation : class
        => value as TActivation
           ?? throw new InvalidOperationException(
               $"Cannot rebuild {layerName}: constructor parameter '{parameterName}' needs " +
               $"a {typeof(TActivation).Name}, but the saved activation was " +
               (value is null ? "not recorded." : $"a {value.GetType().Name}."));



    /// <summary>Records a component's concrete type so it can be rebuilt exactly.</summary>
    /// <param name="value">The component instance, or <c>null</c>.</param>
    /// <returns>The assembly-qualified type name, or empty when there is nothing to record.</returns>
    public static string FormatType(object? value)
        => value is null
            ? string.Empty
            : value.GetType().AssemblyQualifiedName ?? value.GetType().FullName ?? string.Empty;

    private InvalidOperationException Unparseable(string key, object value, string wanted)
        => new($"Cannot rebuild {_layerName}: metadata key '{key}' should be {wanted} but held '{AsText(value)}'.");

    private static string AsText(object value)
        => value as string ?? Convert.ToString(value, CultureInfo.InvariantCulture) ?? string.Empty;

    /// <summary>Formats a value for storage in layer metadata.</summary>
    /// <param name="value">The value to format.</param>
    /// <returns>The invariant-culture text form.</returns>
    public static string Format(int value) => value.ToString(CultureInfo.InvariantCulture);

    /// <inheritdoc cref="Format(int)"/>
    public static string Format(long value) => value.ToString(CultureInfo.InvariantCulture);

    /// <inheritdoc cref="Format(int)"/>
    /// <remarks>Round-trip ("R") so a restored double is bit-identical to the saved one.</remarks>
    public static string Format(double value) => value.ToString("R", CultureInfo.InvariantCulture);

    /// <inheritdoc cref="Format(int)"/>
    public static string Format(float value) => value.ToString("R", CultureInfo.InvariantCulture);

    /// <inheritdoc cref="Format(int)"/>
    public static string Format(bool value) => value ? "true" : "false";

    /// <inheritdoc cref="Format(int)"/>
    public static string Format(string? value) => value ?? string.Empty;

    /// <inheritdoc cref="Format(int)"/>
    public static string Format(Enum value) => value.ToString();

    /// <inheritdoc cref="Format(int)"/>
    public static string Format(int[]? value) => value is null ? string.Empty : string.Join(",", value);

    /// <inheritdoc cref="Format(int)"/>
    public static string Format(bool[]? value) => value is null ? string.Empty : string.Join(",", value);


    /// <inheritdoc cref="Format(int)"/>
    public static string Format(double[]? value)
        => value is null ? string.Empty : string.Join(",", value.Select(d => d.ToString("R", CultureInfo.InvariantCulture)));

    /// <inheritdoc cref="Format(int)"/>
    /// <remarks>Rows joined by ';', values within a row by ',' -- an empty row stays empty.</remarks>
    public static string Format(int[][]? value)
        => value is null ? string.Empty : string.Join(";", value.Select(row => row is null ? string.Empty : string.Join(",", row)));

    /// <summary>Formats a nullable value whose constructor parameter is not itself nullable.</summary>
    /// <typeparam name="TValue">The underlying value type.</typeparam>
    /// <param name="value">The stored value, which may be unset.</param>
    /// <returns>The formatted value, or empty when unset.</returns>
    /// <remarks>
    /// A layer may hold an <c>int</c> constructor argument in an <c>int?</c> field, so the writer
    /// reads back a nullable where the parameter was not. Unset is written as empty rather than as
    /// a stand-in value, so a rebuild reports the key as unparseable instead of quietly restoring
    /// a zero the layer was never given.
    /// </remarks>
    public static string Format<TValue>(TValue? value) where TValue : struct
        => value.HasValue ? Convert.ToString(value.Value, CultureInfo.InvariantCulture) ?? string.Empty : string.Empty;
}
