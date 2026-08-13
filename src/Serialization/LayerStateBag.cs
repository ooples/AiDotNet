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

    /// <summary>Reads an integer array, or <paramref name="fallback"/> when it was not saved.</summary>
    /// <param name="key">The metadata key.</param>
    /// <param name="fallback">Value to use when the key is absent.</param>
    /// <returns>The stored value, or the fallback.</returns>
    public int[]? Int32Array(string key, int[]? fallback) => Has(key) ? Int32Array(key) : fallback;

    // The four accessors below exist for the SAME reason Int32Array does: a layer whose constructor
    // takes an array must be able to read it back, and a factory that cannot read its own saved
    // value would rebuild the layer with a default instead -- silently, because nothing throws when
    // a constructor is handed a plausible wrong argument. They are used by the out-of-assembly
    // factory registry, where the constructor cannot be named at compile time.

    /// <summary>Reads a required double array, stored comma-separated.</summary>
    /// <param name="key">The metadata key.</param>
    /// <returns>The stored value.</returns>
    public double[] DoubleArray(string key)
    {
        if (!TryRaw(key, out var v)) throw Missing(key, "a comma-separated list of numbers");
        if (v is double[] arr) return arr;

        var text = AsText(v);
        if (text.Length == 0) return [];

        var parts = text.Split([',', ' '], StringSplitOptions.RemoveEmptyEntries);
        var result = new double[parts.Length];
        for (int i = 0; i < parts.Length; i++)
        {
            if (!double.TryParse(parts[i], NumberStyles.Float, CultureInfo.InvariantCulture, out result[i]))
                throw Unparseable(key, v, "a comma-separated list of numbers");
        }
        return result;
    }

    /// <summary>Reads a double array, or <paramref name="fallback"/> when it was not saved.</summary>
    /// <param name="key">The metadata key.</param>
    /// <param name="fallback">Value to use when the key is absent.</param>
    /// <returns>The stored value, or the fallback.</returns>
    public double[]? DoubleArray(string key, double[]? fallback) => Has(key) ? DoubleArray(key) : fallback;

    /// <summary>Reads a required boolean array, stored comma-separated.</summary>
    /// <param name="key">The metadata key.</param>
    /// <returns>The stored value.</returns>
    public bool[] BooleanArray(string key)
    {
        if (!TryRaw(key, out var v)) throw Missing(key, "a comma-separated list of true/false");
        if (v is bool[] arr) return arr;

        var text = AsText(v);
        if (text.Length == 0) return [];

        var parts = text.Split([',', ' '], StringSplitOptions.RemoveEmptyEntries);
        var result = new bool[parts.Length];
        for (int i = 0; i < parts.Length; i++)
        {
            if (!bool.TryParse(parts[i], out result[i]))
                throw Unparseable(key, v, "a comma-separated list of true/false");
        }
        return result;
    }

    /// <summary>Reads a boolean array, or <paramref name="fallback"/> when it was not saved.</summary>
    /// <param name="key">The metadata key.</param>
    /// <param name="fallback">Value to use when the key is absent.</param>
    /// <returns>The stored value, or the fallback.</returns>
    public bool[]? BooleanArray(string key, bool[]? fallback) => Has(key) ? BooleanArray(key) : fallback;

    /// <summary>Reads a required string array, stored newline-separated.</summary>
    /// <param name="key">The metadata key.</param>
    /// <returns>The stored value.</returns>
    /// <remarks>
    /// Newline-separated rather than comma-separated: a saved string may legitimately contain a
    /// comma, and splitting on one would turn a single vocabulary entry into two.
    /// </remarks>
    public string[] StringArray(string key)
    {
        if (!TryRaw(key, out var v)) throw Missing(key, "a newline-separated list of strings");
        if (v is string[] arr) return arr;

        var text = AsText(v);
        return text.Length == 0 ? [] : text.Split('\n');
    }

    /// <summary>Reads a string array, or <paramref name="fallback"/> when it was not saved.</summary>
    /// <param name="key">The metadata key.</param>
    /// <param name="fallback">Value to use when the key is absent.</param>
    /// <returns>The stored value, or the fallback.</returns>
    public string[]? StringArray(string key, string[]? fallback) => Has(key) ? StringArray(key) : fallback;

    /// <summary>Reads a required jagged integer array: rows separated by ';', values by ','.</summary>
    /// <param name="key">The metadata key.</param>
    /// <returns>The stored value.</returns>
    public int[][] Int32Jagged(string key)
    {
        if (!TryRaw(key, out var v)) throw Missing(key, "semicolon-separated rows of comma-separated integers");
        if (v is int[][] arr) return arr;

        var text = AsText(v);
        if (text.Length == 0) return [];

        var rows = text.Split([';'], StringSplitOptions.RemoveEmptyEntries);
        var result = new int[rows.Length][];
        for (int r = 0; r < rows.Length; r++)
        {
            var parts = rows[r].Split([',', ' '], StringSplitOptions.RemoveEmptyEntries);
            result[r] = new int[parts.Length];
            for (int c = 0; c < parts.Length; c++)
            {
                if (!int.TryParse(parts[c], NumberStyles.Integer, CultureInfo.InvariantCulture, out result[r][c]))
                    throw Unparseable(key, v, "semicolon-separated rows of comma-separated integers");
            }
        }
        return result;
    }

    /// <summary>Reads a jagged integer array, or <paramref name="fallback"/> when it was not saved.</summary>
    /// <param name="key">The metadata key.</param>
    /// <param name="fallback">Value to use when the key is absent.</param>
    /// <returns>The stored value, or the fallback.</returns>
    public int[][]? Int32Jagged(string key, int[][]? fallback) => Has(key) ? Int32Jagged(key) : fallback;

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
            if (args.Length > 0)
            {
                // MakeGenericType throws ArgumentException on an arity mismatch. Raw, that escapes
                // with neither the layer name nor the key, so a payload naming a two-parameter type
                // where a one-parameter one belongs reports nothing about which layer or field.
                try
                {
                    type = type.MakeGenericType(args);
                }
                catch (ArgumentException ex)
                {
                    throw new InvalidOperationException(
                        $"Cannot rebuild {_layerName}: '{typeName}' was saved for '{key}', but its "
                        + $"generic arity does not match {typeof(TComponent).Name}.", ex);
                }
            }
        }

        // ASSIGNABILITY IS CHECKED BEFORE CONSTRUCTION, not after. typeName comes out of a model
        // file, and Activator.CreateInstance runs the named type's constructor -- so filtering with
        // `as TComponent` afterwards prevented nothing: the object, and any side effect its
        // constructor had, already existed. Any type in any loadable assembly with a public
        // parameterless constructor was reachable from a crafted payload.
        if (!typeof(TComponent).IsAssignableFrom(type))
        {
            throw new InvalidOperationException(
                $"Cannot rebuild {_layerName}: '{key}' names '{typeName}', which is not a "
                + $"{typeof(TComponent).Name}. Substituting a default would silently change what the "
                + "layer computes, so the rebuild fails instead.");
        }

        // A WRONG TYPE THROWS RATHER THAN RETURNING NULL, for the reason this class documents about
        // itself: a caller that receives null applies its own constructor default, and a layer built
        // with a Multiquadric kernel that reloads as the default Gaussian is a different function
        // that nothing downstream would report. The unloadable-type path above already throws; this
        // one now matches it.
        try
        {
            var created = Activator.CreateInstance(type)
                ?? throw new InvalidOperationException(
                    $"Activator returned null for '{type.FullName}'. A component type that cannot be "
                    + "instantiated must fail here rather than hand back a null component that only "
                    + "reports itself much later, as a null reference in unrelated code.");

            return (TComponent)created;
        }
        catch (MissingMethodException ex)
        {
            throw new InvalidOperationException(
                $"Cannot rebuild {_layerName}: '{typeName}' was saved for '{key}', but it has no "
                + "public parameterless constructor to rebuild it with.", ex);
        }
    }

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
}
