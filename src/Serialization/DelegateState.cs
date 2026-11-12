using System.Reflection;
using AiDotNet.Tensors.Engines.Export;
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
/// <item>a portable tensor program (AiDotNet.Tensors <c>TensorProgram</c>), for a <c>Func&lt;Tensor&lt;T&gt;, Tensor&lt;T&gt;&gt;</c>
/// written in engine ops -- closures and captured weights included, and replay can only call engine ops;</item>
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

    /// <summary>
    /// Marks a saved value as a legacy traced computation graph (written by AiDotNet versions before the
    /// legacy autodiff was removed). Still loaded, by translation to engine ops; never written.
    /// </summary>
    public const string GraphScheme = "graph:";

    /// <summary>
    /// Marks a saved value as a portable tensor program: what a <c>Func&lt;Tensor&lt;T&gt;, Tensor&lt;T&gt;&gt;</c>
    /// computed, recorded as AiDotNet.Tensors engine calls (one program per input rank it was traced at).
    /// </summary>
    public const string ProgramScheme = "program:";

    /// <summary>
    /// Describes <paramref name="value"/> well enough to rebuild it, or returns empty when no
    /// description fits. Without an input shape only a named static method can be described.
    /// </summary>
    /// <param name="value">The delegate the layer was constructed with.</param>
    /// <returns>The saved form, or <see cref="string.Empty"/> when it cannot be saved.</returns>
    public static string Save(Delegate? value) => Save(value, null);

    /// <summary>
    /// Describes <paramref name="value"/> well enough to rebuild it, or returns empty when no description fits.
    /// </summary>
    /// <param name="value">The delegate the layer was constructed with.</param>
    /// <param name="inputShape">
    /// The layer's input shape. When given and <paramref name="value"/> is a <c>Func&lt;Tensor&lt;T&gt;, Tensor&lt;T&gt;&gt;</c>,
    /// the function is traced into a portable <see cref="TensorProgram"/>: the only description that survives a
    /// closure (captured weights included), because it records what the delegate DID rather than what it is called.
    /// </param>
    /// <returns>The saved form, or <see cref="string.Empty"/> when it cannot be saved.</returns>
    /// <remarks>
    /// The function is traced at the input shape as given and with a leading batch axis, because layers differ on
    /// whether their input shape includes it; every trace that succeeds is kept and the one matching the input's
    /// rank replays. A function that does part of its work outside the engine fails its trace's verification and
    /// falls through to the method-reference description rather than saving a wrong program.
    /// </remarks>
    public static string Save(Delegate? value, int[]? inputShape)
    {
        if (value is null) return string.Empty;
        if (inputShape is not null && TensorFunctionElementType(value.GetType()) is { } element)
        {
            var saved = (string?)typeof(DelegateState)
                .GetMethod(nameof(SaveProgram), BindingFlags.NonPublic | BindingFlags.Static)!
                .MakeGenericMethod(element)
                .Invoke(null, [value, inputShape]);
            if (saved is not null) return saved;
        }
        return SaveMethodReference(value) ?? string.Empty;
    }

    // T when the type is Func<Tensor<T>, Tensor<T>>.
    private static Type? TensorFunctionElementType(Type delegateType)
    {
        if (!delegateType.IsGenericType || delegateType.GetGenericTypeDefinition() != typeof(Func<,>)) return null;
        var args = delegateType.GetGenericArguments();
        if (args[0] != args[1] || !args[0].IsGenericType || args[0].GetGenericTypeDefinition() != typeof(Tensor<>)) return null;
        return args[0].GetGenericArguments()[0];
    }

    private static string? SaveProgram<T>(Func<Tensor<T>, Tensor<T>> function, int[] inputShape)
    {
        var shape = (inputShape.Length == 0 ? [1] : inputShape).Select(d => d > 0 ? d : 1).ToArray();
        var packages = new List<byte[]>();
        foreach (var probeShape in new[] { shape, new[] { 1 }.Concat(shape).ToArray() })
        {
            try
            {
                var program = TensorProgram.Trace(function, ProbeInput<T>(probeShape));
                using var ms = new MemoryStream();
                new ProgramPackage(program).Save(ms);
                packages.Add(ms.ToArray());
            }
            catch (Exception ex) when (ex is TraceException or ArgumentException or InvalidOperationException
                                           or NotSupportedException or IndexOutOfRangeException)
            {
                // Not traceable at this rank (the function expects the other one, or does work outside the
                // engine): try the next description.
            }
        }
        if (packages.Count == 0) return null;
        using var payload = new MemoryStream();
        using (var w = new BinaryWriter(payload, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            w.Write(packages.Count);
            foreach (var p in packages)
            {
                w.Write(p.Length);
                w.Write(p);
            }
        }
        return ProgramScheme + Convert.ToBase64String(payload.ToArray());
    }

    // Deterministic, non-degenerate probe values in [-1, 1): zeros would hide data-dependent paths and make the
    // replay verification vacuous.
    private static Tensor<T> ProbeInput<T>(int[] shape)
    {
        int length = shape.Aggregate(1, (a, b) => checked(a * b));
        var data = new double[length];
        var rng = new Random(0x0A1D);
        for (int i = 0; i < length; i++) data[i] = rng.NextDouble() * 2 - 1;
        return new Tensor<double>(data, shape).Cast<T>();
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
            .Select(p => p.ParameterType.AssemblyQualifiedName is { } qualifiedName ? qualifiedName : p.ParameterType.FullName ?? string.Empty));

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
                "nothing was recorded for it -- the function could not be traced into a tensor program (it does "
                + "work outside the engine). Write it in engine ops, or build the layer from an expression "
                + "tree, if this layer needs to survive a save.");

        if ((saved ?? throw new InvalidOperationException("saved is null where it is required.")).StartsWith(MethodScheme, StringComparison.Ordinal))
            return LoadMethodReference<TDelegate>(saved.Substring(MethodScheme.Length), layerName, key);
        if (saved.StartsWith(ProgramScheme, StringComparison.Ordinal))
            return LoadTensorFunction<TDelegate>(saved.Substring(ProgramScheme.Length), legacy: false, layerName, key);
        if (saved.StartsWith(GraphScheme, StringComparison.Ordinal))
            return LoadTensorFunction<TDelegate>(saved.Substring(GraphScheme.Length), legacy: true, layerName, key);

        throw Unsaveable(layerName, key, $"its saved form '{Excerpt(saved)}' is not a form this version understands.");
    }

    // A program (or legacy graph) only rebuilds a Func<Tensor<T>, Tensor<T>>; T is read off the delegate type so the
    // generated factory need not thread it through.
    private static TDelegate LoadTensorFunction<TDelegate>(string body, bool legacy, string layerName, string key)
        where TDelegate : Delegate
    {
        var element = TensorFunctionElementType(typeof(TDelegate))
            ?? throw Unsaveable(layerName, key,
                $"it was saved as a tensor program, which only rebuilds a Func<Tensor<T>, Tensor<T>>, not a {typeof(TDelegate).Name}.");
        var method = typeof(DelegateState).GetMethod(
            legacy ? nameof(LoadLegacyGraph) : nameof(LoadPrograms), BindingFlags.NonPublic | BindingFlags.Static)!
            .MakeGenericMethod(element);
        try
        {
            return (TDelegate)method.Invoke(null, [body, layerName, key])!;
        }
        catch (TargetInvocationException ex) when (ex.InnerException is not null)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
            throw;
        }
    }

    private static Func<Tensor<T>, Tensor<T>> LoadLegacyGraph<T>(string graph, string layerName, string key)
        => LegacyGraphTranslator.Compile<T>(graph, layerName, key);

    private static Func<Tensor<T>, Tensor<T>> LoadPrograms<T>(string base64, string layerName, string key)
    {
        byte[] payload;
        try { payload = Convert.FromBase64String(base64); }
        catch (FormatException) { throw Unsaveable(layerName, key, "its saved tensor program is not valid base64."); }

        var programs = new List<TensorProgram>();
        try
        {
            using var r = new BinaryReader(new MemoryStream(payload, writable: false));
            int count = r.ReadInt32();
            if (count <= 0 || count > 8) throw new ProgramFormatException($"a program count of {count} is invalid.");
            for (int i = 0; i < count; i++)
            {
                int length = r.ReadInt32();
                if (length <= 0 || length > payload.Length) throw new ProgramFormatException("a program length is corrupt.");
                programs.Add(ProgramPackage.Load(new MemoryStream(r.ReadBytes(length), writable: false)).Program);
            }
        }
        catch (Exception ex) when (ex is ProgramFormatException or EndOfStreamException)
        {
            throw Unsaveable(layerName, key, $"its saved tensor program could not be read: {ex.Message}");
        }

        return input =>
        {
            if (input is null) throw new ArgumentNullException(nameof(input));
            foreach (var program in programs)
                if (program.InputDimensions[0].Count == input.Rank) return program.Run(input);
            throw new ArgumentException(
                $"{layerName}: the saved function was traced for input rank(s) "
                + $"{string.Join(", ", programs.Select(p => p.InputDimensions[0].Count))}; got rank {input.Rank}.", nameof(input));
        };
    }

    /// <summary>Rebuilds a delegate from a reference to a named static method.</summary>
    private static TDelegate LoadMethodReference<TDelegate>(string reference, string layerName, string key)
        where TDelegate : Delegate
    {
        var parts = reference.Split('|');
        if (parts.Length != 3)
            throw Unsaveable(layerName, key, $"its saved method reference '{Excerpt(reference)}' is malformed.");

        var declaring = Type.GetType(parts[0], throwOnError: false);
        if (declaring is null)
            throw Unsaveable(layerName, key, $"the type '{Excerpt(parts[0])}' that declared it could not be loaded.");

        var parameterTypeNames = parts[2].Length == 0 ? Array.Empty<string>() : parts[2].Split(';');
        var parameterTypes = new Type[parameterTypeNames.Length];
        for (int i = 0; i < parameterTypeNames.Length; i++)
        {
            parameterTypes[i] = Type.GetType(parameterTypeNames[i], throwOnError: false)
                ?? throw Unsaveable(layerName, key, "one of its parameter types could not be loaded.");
        }

        var method = declaring.GetMethod(
            parts[1],
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static,
            binder: null,
            types: parameterTypes,
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
