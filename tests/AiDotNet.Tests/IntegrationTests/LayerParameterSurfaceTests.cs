using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests;

/// <summary>
/// Asserts, for every constructable layer, that <c>ParameterCount</c> equals
/// <c>GetParameters().Length</c>.
/// </summary>
/// <remarks>
/// <para>
/// This is the invariant the MODEL-level failures are downstream of.
/// <c>NeuralNetworkBase.ParameterCount</c> sums <c>layer.ParameterCount</c> while
/// <c>NeuralNetworkBase.GetParameters()</c> sums <c>layer.GetParameters().Length</c> — the same
/// list, but a DIFFERENT question asked of each layer. Wherever one layer answers those two
/// questions differently, every model containing it reports a mismatch, and the model looks like
/// the culprit. Three of the models failing CI (MusicSourceSeparator, RoomImpulseResponse,
/// OpenVoiceV2) override neither member, which is the proof: the divergence is underneath them.
/// </para>
/// <para>
/// PyTorch cannot express this bug. <c>nn.Module.parameters()</c> is the sole registry, populated
/// automatically by <c>__setattr__</c> interception, and the count is
/// <c>sum(p.numel() for p in model.parameters())</c> — a fold over the very tensors the iterator
/// yields. There is no per-module count to drift from it. The equivalent guarantee here is that
/// each layer's two surfaces describe one set of tensors; once that holds, a model summing either
/// one gets the same answer, and models stop needing to hand-roll agreement between them.
/// </para>
/// <para>
/// Reports every violation in one message rather than failing at the first, for the same reason
/// the model sweep does: fixing one and re-running tells you nothing about how many remain.
/// </para>
/// </remarks>
[Collection("ParameterSweeps")]
[Trait("Category", "Sweep")]
public class LayerParameterSurfaceTests
{
    private readonly ITestOutputHelper _output;

    public LayerParameterSurfaceTests(ITestOutputHelper output) => _output = output;

    [Fact(Timeout = 900000)]
    public async System.Threading.Tasks.Task AllLayers_ParameterCountMatchesGetParameters()
    {
        await System.Threading.Tasks.Task.Yield();

        var violations = new List<string>();
        // EVERY VISITED LAYER LANDS IN A BUCKET. A layer with no public GetParameters() used to
        // be skipped with a bare continue, incrementing nothing, so the summary reported a total
        // that silently omitted it -- inflating coverage in exactly the way the counting exists
        // to prevent.
        int checkedCount = 0, unconstructable = 0, unsized = 0, noParameterApi = 0;
        int warmedUp = 0, notWarmedUp = 0;

        var logPath = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "layer-parameter-surface.txt");
        // DISPOSED ON EVERY PATH, AND A FAILURE TO OPEN IS REPORTED. The manual dispose ran only
        // on the normal path, so a throw from the enumeration below -- Assembly.GetTypes() raises
        // ReflectionTypeLoadException, and this sweep calls it -- leaked the handle and left the
        // partial log the comment above calls "the deliverable" unflushed. The empty catch was the
        // other half: when the file could not be opened at all, every later write silently went
        // nowhere and the run looked like one that simply found nothing.
        System.IO.StreamWriter? log = null;
        try
        {
            log = new System.IO.StreamWriter(logPath, append: false) { AutoFlush = true };
        }
        catch (Exception ex)
        {
            _output.WriteLine($"NOTE: could not open {logPath} ({ex.GetType().Name}: {ex.Message}); " +
                              "the console output below is the only record of this run.");
        }

        using var _logHandle = log;

        foreach (var closed in GetConstructableLayerTypes())
        {
            var name = closed.FullName ?? closed.Name;
            log?.WriteLine($"[measuring] {name}");

            object? layer;
            try { layer = TryConstruct(closed); }
            catch { layer = null; }
            if (layer is null) { unconstructable++; continue; }

            // Drive one forward before measuring. A layer built from its declared arguments alone
            // knows its OUTPUT width and nothing else, so it sits shape-deferred with no weights
            // allocated -- and 0 == 0 satisfies this invariant without testing anything. Sixty-odd
            // weight-holding layers passed that way: LSTM, GRU, Attention, BatchNormalization, the
            // convolutions, the transformer blocks. The declared TestInputShape is the missing half
            // of the same metadata the constructor arguments come from, so feeding it is what turns
            // those into real measurements.
            bool warm = TryWarmUp(closed, layer);
            if (warm) warmedUp++;
            else notWarmedUp++;

            try
            {
                long declared = Convert.ToInt64(
                    closed.GetProperty("ParameterCount")!.GetValue(layer));

                // A layer whose shape is still deferred legitimately reports nothing from BOTH
                // surfaces; that is agreement, not a violation, and it is the state the deferred
                // layers were fixed INTO. Counted so the coverage is honest rather than inflated.
                var uninitProp = closed.GetProperty("HasUninitializedParameters",
                    BindingFlags.Public | BindingFlags.Instance | BindingFlags.FlattenHierarchy);
                bool pending = uninitProp?.GetValue(layer) is true;

                var m = closed.GetMethod("GetParameters", BindingFlags.Public | BindingFlags.Instance,
                    null, Type.EmptyTypes, null);
                if (m is null) { noParameterApi++; continue; }
                var vec = m.Invoke(layer, null);
                long actual = vec is null ? 0
                    : Convert.ToInt64(vec.GetType().GetProperty("Length")!.GetValue(vec));

                checkedCount++;
                if (declared == actual)
                {
                    // The AGREED value, not just the fact of agreement. A layer can be moved out of
                    // the violation list by making both surfaces report NOTHING -- suppressing the
                    // fallback that was materializing its children is enough -- and that reads
                    // identically to a real fix unless the number is written down. Recording it is
                    // what lets "agrees at 592" be told apart from "agrees at 0".
                    // Warm status on the line, because a zero means two different things. A layer
                    // that RAN a forward and still holds nothing is parameter-free; one that could
                    // not be driven is merely untested, and telling them apart is the whole point
                    // of warming up at all.
                    log?.WriteLine($"AGREE {name}: {declared}" +
                                   $"{(pending ? " [deferred]" : "")}{(warm ? "" : " [not warmed]")}");
                    if (pending) unsized++;
                    continue;
                }

                var row = $"{name}: ParameterCount={declared}, GetParameters().Length={actual} " +
                          $"(difference {declared - actual}){(pending ? " [deferred]" : "")}";
                violations.Add(row);
                log?.WriteLine("VIOLATION " + row);
            }
            catch (Exception ex)
            {
                unconstructable++;
                _output.WriteLine($"UNMEASURABLE {name}: {ex.GetBaseException().GetType().Name}");
            }
        }

        _output.WriteLine($"Checked {checkedCount} layers; {unsized} agree at zero (deferred); " +
                          $"{noParameterApi} expose no public GetParameters(); " +
                          $"{unconstructable} not constructable; {warmedUp} warmed up, " +
                          $"{notWarmedUp} measured without a forward; {violations.Count} violations.");
        foreach (var v in violations.OrderBy(v => v, StringComparer.Ordinal))
            _output.WriteLine("  " + v);

        Assert.True(violations.Count == 0,
            $"{violations.Count} layer(s) report a ParameterCount that disagrees with " +
            "GetParameters().Length.\n\n" +
            "Every model containing such a layer reports a mismatch of its own, because " +
            "NeuralNetworkBase.ParameterCount sums layer.ParameterCount while GetParameters() sums " +
            "layer.GetParameters().Length. Fixing the layer fixes every model that holds it; " +
            "fixing the models one at a time does not.\n\n" +
            string.Join("\n", violations.OrderBy(v => v, StringComparer.Ordinal).Select(v => "  " + v)));
    }

    /// <summary>
    /// Builds a layer for measurement, preferring a default constructor and falling back to the
    /// constructor arguments the layer already declares for scaffold generation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Requiring a parameterless constructor reached 39 layers. Every composite this invariant is
    /// actually about — ClozeAttention, BranchformerBlock, ConformerBlock, VideoGigaGAN, the ResNet
    /// blocks — takes its widths as constructor arguments, so all of them sat outside the sweep and
    /// it reported agreement across a set that excluded them. That is the worse failure: a green
    /// sweep over the layers that were never at risk reads exactly like a green sweep over all of
    /// them.
    /// </para>
    /// <para>
    /// The widths are not guessed here. <c>[LayerProperty(TestConstructorArgs = "...")]</c> already
    /// states them on the layer, and the test scaffold generator emits real constructor calls from
    /// that same string; this replays it through reflection.
    /// </para>
    /// <para>
    /// The declaration is C# source, so replaying it means evaluating the small expression language
    /// it actually uses: numbers, <c>null</c> behind a cast, <c>true</c>/<c>false</c>, <c>new[] { 1,
    /// 4 }</c> and its jagged form, enum members, nested <c>new SomeLayer&lt;double&gt;(...)</c>, and
    /// named arguments. Anything outside that — an object initializer, say — leaves the layer
    /// unmeasured and COUNTED as unconstructable, because a sweep that silently dropped it would
    /// report the same clean result whether the layer agreed or was simply never asked.
    /// </para>
    /// </remarks>
    private static object? TryConstruct(Type closed)
    {
        var ctor = closed.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
            .Where(c => c.GetParameters().All(p => p.HasDefaultValue) || c.GetParameters().Length == 0)
            .OrderBy(c => c.GetParameters().Length)
            .FirstOrDefault();
        if (ctor is not null)
        {
            var defaults = ctor.GetParameters()
                .Select(p => p.DefaultValue == DBNull.Value ? null : p.DefaultValue).ToArray();
            return ctor.Invoke(defaults);
        }

        var declared = TestConstructorArgs(closed);
        return string.IsNullOrWhiteSpace(declared) ? null : Instantiate(closed, declared);
    }

    /// <summary>
    /// Runs one forward from the layer's declared <c>TestInputShape</c>, so its weights exist by
    /// the time the two surfaces are compared. False when the layer declares no usable shape or
    /// refuses the input.
    /// </summary>
    /// <remarks>
    /// Reported rather than required. A layer that cannot be driven is still measured, in whatever
    /// state it reached -- the alternative, skipping it, would shrink the denominator to the layers
    /// that happened to cooperate and call the result full coverage.
    /// </remarks>
    private static bool TryWarmUp(Type closed, object layer)
    {
        var declared = closed.GetCustomAttributes(inherit: false)
            .OfType<AiDotNet.Attributes.LayerPropertyAttribute>()
            .FirstOrDefault()?.TestInputShape;
        if (string.IsNullOrWhiteSpace(declared)) return false;

        var dims = new List<int>();
        foreach (var token in declared!.Split(','))
        {
            if (!int.TryParse(token.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture,
                    out int dim) || dim <= 0)
                return false;
            dims.Add(dim);
        }
        if (dims.Count == 0) return false;

        // Pick the TENSOR overload by its parameter type. LayerBase declares three one-argument
        // Forwards -- Tensor, params Tensor[], and IReadOnlyDictionary -- and taking whichever
        // reflection listed first selected a non-tensor one for every layer in the library, so the
        // warm-up silently did nothing at all: 0 of 210 warmed up.
        var forward = closed.GetMethods(BindingFlags.Public | BindingFlags.Instance)
            .FirstOrDefault(m => m.Name == "Forward"
                && m.GetParameters().Length == 1
                && m.GetParameters()[0].ParameterType.IsGenericType
                && m.GetParameters()[0].ParameterType.GetGenericTypeDefinition().Name
                    .StartsWith("Tensor", StringComparison.Ordinal));
        if (forward is null) return false;

        var tensorType = forward.GetParameters()[0].ParameterType;

        try
        {
            var input = Activator.CreateInstance(tensorType, new object[] { dims.ToArray() });
            if (input is null) return false;
            forward.Invoke(layer, new[] { input });
            return true;
        }
        catch (Exception)
        {
            // A layer may need several inputs, a mask, or a shape this one declaration does not
            // describe. It stays measured; it just stays deferred.
            return false;
        }
    }

    /// <summary>True when the layer states constructor arguments for scaffold generation.</summary>
    private static bool DeclaresTestArguments(Type closed)
        => !string.IsNullOrWhiteSpace(TestConstructorArgs(closed));

    private static string? TestConstructorArgs(Type closed)
        => closed.GetCustomAttributes(inherit: false)
            .OfType<AiDotNet.Attributes.LayerPropertyAttribute>()
            .FirstOrDefault()?.TestConstructorArgs;

    /// <summary>Builds <paramref name="type"/> from a C# argument list, or null if it cannot.</summary>
    private static object? Instantiate(Type type, string argumentList)
    {
        var declared = SplitArguments(argumentList);
        if (declared is null) return null;

        foreach (var candidate in type.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
                     .OrderBy(c => c.GetParameters().Length))
        {
            if (TryBind(candidate, declared, out var args)) return candidate.Invoke(args);
        }
        return null;
    }

    /// <summary>
    /// Matches one declared argument list against one constructor overload, evaluating each
    /// argument into the parameter it lands in.
    /// </summary>
    /// <remarks>
    /// A named argument is bound BY NAME rather than by position. Stripping the name and taking the
    /// slot it happened to sit in would quietly build a different layer than the declaration
    /// describes, and the sweep would then measure that one and report on it as though it were the
    /// declared configuration.
    /// </remarks>
    private static bool TryBind(
        ConstructorInfo candidate,
        IReadOnlyList<(string? Name, string Expression)> declared,
        out object?[] args)
    {
        var parameters = candidate.GetParameters();
        args = new object?[parameters.Length];
        var bound = new bool[parameters.Length];

        int position = 0;
        foreach (var (name, expression) in declared)
        {
            int index = name is null
                ? position++
                : Array.FindIndex(parameters, p => string.Equals(p.Name, name, StringComparison.Ordinal));
            if (index < 0 || index >= parameters.Length || bound[index]) return false;
            if (!TryEvaluate(expression, out object? value)) return false;
            if (!TryCoerce(value, parameters[index].ParameterType, out object? coerced)) return false;
            args[index] = coerced;
            bound[index] = true;
        }

        for (int i = 0; i < parameters.Length; i++)
        {
            if (bound[i]) continue;
            if (!parameters[i].HasDefaultValue) return false;
            args[i] = parameters[i].DefaultValue == DBNull.Value ? null : parameters[i].DefaultValue;
        }
        return true;
    }

    /// <summary>
    /// Splits an argument list on its top-level commas, keeping <c>new[] { 1, 4 }</c> whole, and
    /// separating any <c>name:</c> prefix. Null when the text is unbalanced.
    /// </summary>
    private static IReadOnlyList<(string? Name, string Expression)>? SplitArguments(string text)
    {
        var pieces = SplitTopLevel(text);
        if (pieces is null) return null;

        var declared = new List<(string? Name, string Expression)>(pieces.Count);
        foreach (var piece in pieces)
        {
            var trimmed = piece.Trim();
            if (trimmed.Length == 0) return null;

            int colon = TopLevelColon(trimmed);
            declared.Add(colon < 0
                ? (null, trimmed)
                : (trimmed.Substring(0, colon).Trim(), trimmed.Substring(colon + 1).Trim()));
        }
        return declared;
    }

    private static List<string>? SplitTopLevel(string text)
    {
        var pieces = new List<string>();
        int depth = 0, start = 0;
        bool inString = false;
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (inString)
            {
                if (c == '"') inString = false;
                continue;
            }
            if (c == '"') { inString = true; continue; }
            if (c is '(' or '{' or '[' or '<') depth++;
            else if (c is ')' or '}' or ']' or '>') depth--;
            else if (c == ',' && depth == 0)
            {
                pieces.Add(text.Substring(start, i - start));
                start = i + 1;
            }
        }
        if (inString || depth != 0) return null;
        pieces.Add(text.Substring(start));
        return pieces;
    }

    /// <summary>Index of a <c>name:</c> separator, skipping <c>::</c> and any nested text.</summary>
    private static int TopLevelColon(string text)
    {
        int depth = 0;
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (c is '(' or '{' or '[' or '<') depth++;
            else if (c is ')' or '}' or ']' or '>') depth--;
            else if (c == ':' && depth == 0)
            {
                bool qualifier = (i > 0 && text[i - 1] == ':') || (i + 1 < text.Length && text[i + 1] == ':');
                if (!qualifier) return i;
            }
        }
        return -1;
    }

    /// <summary>Evaluates one declared argument expression.</summary>
    private static bool TryEvaluate(string expression, out object? value)
    {
        value = null;
        string text = StripCasts(expression);
        if (text.Length == 0) return false;

        if (text == "null") return true;
        if (text == "true") { value = true; return true; }
        if (text == "false") { value = false; return true; }
        if (double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double number))
        {
            value = number;
            return true;
        }
        return text.StartsWith("new", StringComparison.Ordinal)
            ? TryEvaluateNew(text, out value)
            : TryEvaluateEnumMember(text, out value);
    }

    /// <summary>
    /// Removes leading casts. The declarations use them to pick an overload — <c>(IActivationFunction
    /// &lt;double&gt;?)null</c> — which reflection resolves from the parameter type instead.
    /// </summary>
    private static string StripCasts(string expression)
    {
        string text = expression.Trim();
        while (text.StartsWith("(", StringComparison.Ordinal))
        {
            int close = MatchingBracket(text, 0);
            if (close < 0 || close == text.Length - 1) break;
            string inner = text.Substring(1, close - 1).Trim();
            // A cast names a type; a parenthesised value does not. Requiring a letter and rejecting
            // anything with a top-level comma keeps `(a, b)` and `(3)` from being mistaken for one.
            if (inner.Length == 0 || !inner.Any(char.IsLetter) || inner.Contains(',')) break;
            text = text.Substring(close + 1).Trim();
        }
        return text;
    }

    private static int MatchingBracket(string text, int open)
    {
        int depth = 0;
        for (int i = open; i < text.Length; i++)
        {
            if (text[i] is '(' or '{' or '[') depth++;
            else if (text[i] is ')' or '}' or ']')
            {
                depth--;
                if (depth == 0) return i;
            }
        }
        return -1;
    }

    /// <summary>Evaluates <c>new[] { ... }</c>, <c>new T[] { ... }</c> and <c>new T(...)</c>.</summary>
    private static bool TryEvaluateNew(string text, out object? value)
    {
        value = null;
        string body = text.Substring(3).Trim();

        int brace = body.IndexOf('{');
        int paren = body.IndexOf('(');
        bool isArray = brace >= 0 && (paren < 0 || brace < paren);

        if (isArray)
        {
            // An object initializer -- `new T { Property = ... }` -- also opens with a brace. The
            // array forms are the ones whose text before it is empty or ends in `[]`.
            string prefix = body.Substring(0, brace).Trim();
            if (prefix.Length != 0 && !prefix.EndsWith("[]", StringComparison.Ordinal)) return false;

            int close = MatchingBracket(body, brace);
            if (close < 0) return false;
            var elements = SplitTopLevel(body.Substring(brace + 1, close - brace - 1));
            if (elements is null) return false;

            var evaluated = new List<object?>();
            foreach (var element in elements)
            {
                if (element.Trim().Length == 0) continue;
                if (!TryEvaluate(element, out object? item)) return false;
                evaluated.Add(item);
            }

            if (evaluated.All(item => item is double))
            {
                value = evaluated.Select(item => (int)(double)item!).ToArray();
                return true;
            }
            if (evaluated.All(item => item is int[]))
            {
                value = evaluated.Select(item => (int[])item!).ToArray();
                return true;
            }
            return false;
        }

        if (paren < 0) return false;
        int end = MatchingBracket(body, paren);
        if (end < 0) return false;

        var type = ResolveType(body.Substring(0, paren));
        if (type is null) return false;

        string arguments = body.Substring(paren + 1, end - paren - 1).Trim();
        object? instance = arguments.Length == 0
            ? Activator.CreateInstance(type)
            : Instantiate(type, arguments);
        if (instance is null) return false;
        value = instance;
        return true;
    }

    private static bool TryEvaluateEnumMember(string text, out object? value)
    {
        value = null;
        int dot = text.LastIndexOf('.');
        if (dot <= 0) return false;

        var type = ResolveType(text.Substring(0, dot));
        if (type is null || !type.IsEnum) return false;

        string member = text.Substring(dot + 1).Trim();
        if (!Enum.IsDefined(type, member)) return false;
        value = Enum.Parse(type, member);
        return true;
    }

    /// <summary>
    /// Resolves a source-form type name. Every generic in these declarations is closed over
    /// <c>double</c>, which is also the element type this sweep measures.
    /// </summary>
    private static Type? ResolveType(string name)
    {
        string text = name.Replace("global::", string.Empty).Trim();

        var segments = new List<string>();
        int depth = 0, start = 0;
        for (int i = 0; i < text.Length; i++)
        {
            if (text[i] == '<') depth++;
            else if (text[i] == '>') depth--;
            else if (text[i] == '.' && depth == 0)
            {
                segments.Add(text.Substring(start, i - start));
                start = i + 1;
            }
        }
        segments.Add(text.Substring(start));

        var assembly = typeof(AiDotNet.Models.ModelMetadata<>).Assembly;

        // Longest namespace-qualified prefix first, then shorter ones: the trailing segments become
        // NESTED types, which is how `SomeLayer<double>.Position` has to be spelled to reflection.
        for (int split = segments.Count; split >= 1; split--)
        {
            var builder = new System.Text.StringBuilder();
            for (int i = 0; i < split; i++)
            {
                if (i > 0) builder.Append('.');
                string segment = segments[i].Trim();
                int angle = segment.IndexOf('<');
                if (angle < 0) builder.Append(segment);
                else builder.Append(segment, 0, angle).Append("`1");
            }
            for (int i = split; i < segments.Count; i++) builder.Append('+').Append(segments[i].Trim());

            string candidateName = builder.ToString();
            var candidate = assembly.GetType(candidateName, throwOnError: false)
                            ?? Type.GetType(candidateName, throwOnError: false);
            if (candidate is null) continue;

            return candidate.IsGenericTypeDefinition
                ? candidate.MakeGenericType(typeof(double))
                : candidate;
        }
        return null;
    }

    /// <summary>Fits an evaluated value to the parameter it was declared for.</summary>
    private static bool TryCoerce(object? value, Type target, out object? result)
    {
        result = null;
        var underlying = Nullable.GetUnderlyingType(target) ?? target;

        if (value is null) return !underlying.IsValueType || Nullable.GetUnderlyingType(target) is not null;
        if (underlying.IsInstanceOfType(value)) { result = value; return true; }

        // Numbers arrive as double because the declaration does not say which width it meant.
        // bool and char are primitives too, and turning a declared 1 into true would build a
        // different layer than the declaration describes.
        if (value is double number && underlying.IsPrimitive
            && underlying != typeof(bool) && underlying != typeof(char))
        {
            try
            {
                result = Convert.ChangeType(number, underlying, CultureInfo.InvariantCulture);
                return true;
            }
            catch (Exception ex) when (ex is InvalidCastException or OverflowException or FormatException)
            {
                return false;
            }
        }
        return false;
    }

    private static IEnumerable<Type> GetConstructableLayerTypes()
    {
        var assembly = typeof(AiDotNet.Models.ModelMetadata<>).Assembly;
        foreach (var open in assembly.GetTypes()
                     .Where(t => t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition)
                     .Where(t => t.GetGenericArguments().Length == 1))
        {
            Type closed;
            try { closed = open.MakeGenericType(typeof(double)); } catch { continue; }

            // LayerBase<T> descendants only — this is about the layer contract, not models.
            bool isLayer = false;
            for (var b = closed.BaseType; b is not null; b = b.BaseType)
                if (b.IsGenericType && b.GetGenericTypeDefinition().Name.StartsWith("LayerBase", StringComparison.Ordinal))
                { isLayer = true; break; }
            if (!isLayer) continue;

            // Gated on DECLARING test arguments, not on their being replayable. A layer whose
            // declaration this sweep cannot reconstruct is a coverage MISS, and it has to land in
            // the "not constructable" bucket to be visible as one; filtering it out here instead
            // would drop it from the denominator, so the summary would report full coverage of a
            // set that had quietly shrunk to the layers that happened to be easy.
            if (!closed.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
                    .Any(c => c.GetParameters().Length == 0 || c.GetParameters().All(p => p.HasDefaultValue))
                && !DeclaresTestArguments(closed))
                continue;

            yield return closed;
        }
    }
}
