using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// The verdict of one option-sensitivity measurement.
/// </summary>
/// <remarks>
/// Only <see cref="Sensitive"/> passes. Every other value fails the generated test unless
/// <see cref="RuntimeFidelityBaseline.OptionSensitivity"/> lists the same model, option and outcome,
/// in which case it is reported as a known, skipped defect. The non-<see cref="Insensitive"/> values
/// say why the question could not be answered, so a baseline entry records the actual reason rather
/// than lumping harness limits together with ignored options.
/// </remarks>
public enum OptionSensitivityOutcome
{
    /// <summary>Perturbing the option changed parameter count, parameters or outputs.</summary>
    Sensitive,

    /// <summary>The perturbed option reached the rebuilt model and nothing observable changed: the defect this test exists for.</summary>
    Insensitive,

    /// <summary>The fixture's model holds no instance of its declared options type, so there is nothing to perturb.</summary>
    OptionsNotHeld,

    /// <summary>The model's clone plan does not pass the options instance back into its constructor, so a perturbed copy never reaches a rebuilt model.</summary>
    OptionsNotRouted,

    /// <summary>Every candidate perturbation was rejected when the model was rebuilt.</summary>
    NoValidPerturbation,

    /// <summary>Two identical, identically seeded builds disagreed, so a difference could not be attributed to the option.</summary>
    NonDeterministic,

    /// <summary>The option is not a public writable property of the runtime options type.</summary>
    UnknownOption,
}

/// <summary>
/// How a family base exposes its model to the option-sensitivity harness: how to build it, what to
/// observe, and how to train it for a few fixed steps.
/// </summary>
/// <typeparam name="TModel">The family's model interface.</typeparam>
public sealed class OptionSensitivityProbe<TModel> where TModel : class
{
    /// <summary>Creates the probe.</summary>
    /// <param name="create">The fixture's own factory (CreateModel / CreateNetwork).</param>
    /// <param name="observe">A content fingerprint of the model's parameters and its outputs on a fixed input.</param>
    /// <param name="train">Runs the same fixed, deterministic training steps on any model it is given.</param>
    /// <param name="prepareOptions">
    /// Applied to both the control and the perturbed options copy before the perturbation, for
    /// gates that would otherwise keep a few training steps from reaching an update (an RL warmup).
    /// </param>
    public OptionSensitivityProbe(
        Func<TModel> create,
        Func<TModel, long> observe,
        Action<TModel>? train,
        Action<ModelOptions>? prepareOptions = null)
    {
        Create = create ?? throw new ArgumentNullException(nameof(create));
        Observe = observe ?? throw new ArgumentNullException(nameof(observe));
        Train = train;
        PrepareOptions = prepareOptions;
    }

    /// <summary>The fixture's factory.</summary>
    public Func<TModel> Create { get; }

    /// <summary>The observation fingerprint.</summary>
    public Func<TModel, long> Observe { get; }

    /// <summary>The fixed training steps, or null when the family cannot train generically.</summary>
    public Action<TModel>? Train { get; }

    /// <summary>The options preparation shared by control and perturbed builds.</summary>
    public Action<ModelOptions>? PrepareOptions { get; }
}

/// <summary>The result of measuring one option on one model.</summary>
public sealed class OptionSensitivityReport
{
    internal OptionSensitivityReport(string modelKey, string option, OptionSensitivityOutcome outcome, string detail)
    {
        ModelKey = modelKey;
        Option = option;
        Outcome = outcome;
        Detail = detail;
    }

    /// <summary>The model's generic type definition, as used to key the runtime baseline.</summary>
    public string ModelKey { get; }

    /// <summary>The option measured.</summary>
    public string Option { get; }

    /// <summary>The verdict.</summary>
    public OptionSensitivityOutcome Outcome { get; }

    /// <summary>What was observed, for the failure or skip message.</summary>
    public string Detail { get; }

    /// <summary>The runtime baseline key: <c>ModelKey::Option</c>.</summary>
    public string BaselineKey => ModelKey + "::" + Option;
}

/// <summary>
/// Measures whether perturbing one options property changes a model's observable behaviour.
/// </summary>
/// <remarks>
/// <para>
/// <b>How the perturbed model is built.</b> The fixture's own factory builds the model, so every
/// fixture keeps its test-scale configuration. The options instance the model holds is replaced by an
/// independent copy, the property under test is changed on that copy, and the model is rebuilt through
/// <see cref="CloneEngine.CopyConfiguration"/> - the library's single reconstruction path, which calls
/// the model's constructor with the values its compile-time clone plan binds, including the options.
/// The control goes through exactly the same path with the property left alone, so the two builds
/// differ in that one property and nothing else. Both are built under the same fixed layer
/// initialisation seed, and a null <c>Seed</c> option is pinned on both copies.
/// </para>
/// <para>
/// <b>What is observed.</b> Parameter count, every parameter value, and the outputs on one fixed
/// input, as a single content fingerprint. When those match, both models take the same fixed training
/// steps and are fingerprinted again, which is what exposes options that act only during training
/// (a learning rate, a discount factor). Two control builds are compared first; if they disagree the
/// verdict is <see cref="OptionSensitivityOutcome.NonDeterministic"/>, never a guess.
/// </para>
/// <para>
/// <b>Perturbations.</b> A valid different value of the same type: a number doubled, else incremented,
/// else halved or decremented; a flag flipped; an enum moved to its next member; a list lengthened by
/// repeating its last element, else shortened. Candidates are tried in that order and the first one the
/// model's constructor accepts is used, which is how "within its validation range" is honoured without
/// duplicating every model's validation here.
/// </para>
/// </remarks>
public static class OptionSensitivityHarness
{
    /// <summary>The layer-initialisation seed both builds share.</summary>
    public const int FixedSeed = 1337;

    private static readonly ConcurrentDictionary<Type, Lazy<ControlFingerprint>> s_controls = new();

    /// <summary>
    /// Measures <paramref name="option"/> on the fixture's model and enforces the verdict against the
    /// runtime baseline.
    /// </summary>
    /// <typeparam name="TModel">The family's model interface.</typeparam>
    /// <param name="fixtureType">The concrete test class, used to cache the control measurement.</param>
    /// <param name="option">The options property to perturb.</param>
    /// <param name="probe">How the family builds, observes and trains its model.</param>
    public static void AssertChangesObservableBehaviour<TModel>(Type fixtureType, string option, OptionSensitivityProbe<TModel> probe)
        where TModel : class
    {
        var report = Measure(fixtureType, option, probe);
        Enforce(report);
    }

    /// <summary>Applies the runtime baseline to a measurement: pass, skip as known, or fail.</summary>
    public static void Enforce(OptionSensitivityReport report)
    {
        bool baselined = RuntimeFidelityBaseline.OptionSensitivity.TryGetValue(report.BaselineKey, out var expected);
        if (report.Outcome == OptionSensitivityOutcome.Sensitive)
        {
            Assert.False(baselined,
                $"'{report.BaselineKey}' is listed in RuntimeFidelityBaseline.OptionSensitivity as {expected}, but perturbing "
                + $"it now changes observable behaviour ({report.Detail}). Remove its entry so the baseline stays an exact inventory.");
            return;
        }

        Skip.If(baselined && expected == report.Outcome,
            $"Known defect, listed in RuntimeFidelityBaseline.OptionSensitivity: {report.Outcome}. {report.Detail}");

        string listed = baselined
            ? $" The baseline lists this option as {expected}, so the defect has changed shape; update or remove its entry."
            : string.Empty;
        Assert.Fail($"Option '{report.Option}' of {report.ModelKey}: {report.Outcome}. {report.Detail}{listed}");
    }

    /// <summary>Measures one option without enforcing anything.</summary>
    public static OptionSensitivityReport Measure<TModel>(Type fixtureType, string option, OptionSensitivityProbe<TModel> probe)
        where TModel : class
    {
        if (fixtureType is null) throw new ArgumentNullException(nameof(fixtureType));
        if (probe is null) throw new ArgumentNullException(nameof(probe));

        string modelKey;
        Type declaredOptions;
        PropertyInfo? property;
        object? fixtureValue;
        var source = Seeded(probe.Create);
        try
        {
            modelKey = ModelKey(source.GetType());
            var declared = FindDeclaredOptionsType(source.GetType());
            var held = declared is null ? null : FindHeldOptions(source, declared);
            if (declared is null || held is null)
            {
                return new OptionSensitivityReport(modelKey, option, OptionSensitivityOutcome.OptionsNotHeld,
                    declared is null
                        ? "The model has no public constructor taking a ModelOptions-derived parameter."
                        : $"No field of the model holds an instance of {declared.Name}; it cannot be perturbed.");
            }

            declaredOptions = declared;
            property = FindWritableProperty(held.GetType(), option);
            fixtureValue = property?.GetValue(held);
        }
        finally
        {
            (source as IDisposable)?.Dispose();
        }

        if (property is null)
        {
            return new OptionSensitivityReport(modelKey, option, OptionSensitivityOutcome.UnknownOption,
                $"{declaredOptions.Name} has no public writable property '{option}'.");
        }

        var control = s_controls.GetOrAdd(fixtureType,
            _ => new Lazy<ControlFingerprint>(() => MeasureControl(probe, declaredOptions))).Value;
        if (control.Failure is not null)
        {
            return new OptionSensitivityReport(modelKey, option, control.FailureOutcome, control.Failure);
        }

        var rejected = new List<string>();
        foreach (var candidate in Perturbations(property.PropertyType, fixtureValue))
        {
            TModel? perturbed;
            try
            {
                perturbed = Rebuild(probe, declaredOptions, options => property.SetValue(options, candidate));
            }
            catch (Exception ex) when (IsConstructionRejection(ex))
            {
                rejected.Add($"{Describe(candidate)} ({Unwrap(ex).GetType().Name}: {Unwrap(ex).Message})");
                continue;
            }

            try
            {
                var routed = FindHeldOptions(perturbed, declaredOptions);
                if (routed is null || !ValuesEqual(property.GetValue(routed), candidate))
                {
                    return new OptionSensitivityReport(modelKey, option, OptionSensitivityOutcome.OptionsNotRouted,
                        $"The rebuilt model does not hold {option} = {Describe(candidate)}: its clone plan does not pass the "
                        + "options instance back into the constructor, so no perturbation can reach it.");
                }

                long untrained;
                try
                {
                    untrained = probe.Observe(perturbed);
                }
                catch (Exception ex) when (ex is not OutOfMemoryException)
                {
                    return new OptionSensitivityReport(modelKey, option, OptionSensitivityOutcome.Sensitive,
                        $"with {option} = {Describe(candidate)} the model rejects the fixture input "
                        + $"({Unwrap(ex).GetType().Name}), so the option changes its input contract.");
                }

                if (untrained != control.Untrained)
                {
                    return new OptionSensitivityReport(modelKey, option, OptionSensitivityOutcome.Sensitive,
                        $"{option} {Describe(fixtureValue)} -> {Describe(candidate)} changes parameters or outputs before training.");
                }

                if (probe.Train is null || control.Trained is null)
                {
                    return new OptionSensitivityReport(modelKey, option, OptionSensitivityOutcome.Insensitive,
                        $"{option} {Describe(fixtureValue)} -> {Describe(candidate)} reached the rebuilt model and changed neither its "
                        + "parameters nor its outputs; this family has no generic training step to look further.");
                }

                long trained;
                try
                {
                    probe.Train(perturbed);
                    trained = probe.Observe(perturbed);
                }
                catch (Exception ex) when (ex is not OutOfMemoryException)
                {
                    return new OptionSensitivityReport(modelKey, option, OptionSensitivityOutcome.Sensitive,
                        $"with {option} = {Describe(candidate)} training fails where the control trains "
                        + $"({Unwrap(ex).GetType().Name}: {Unwrap(ex).Message}).");
                }

                return trained != control.Trained.Value
                    ? new OptionSensitivityReport(modelKey, option, OptionSensitivityOutcome.Sensitive,
                        $"{option} {Describe(fixtureValue)} -> {Describe(candidate)} changes parameters or outputs after training.")
                    : new OptionSensitivityReport(modelKey, option, OptionSensitivityOutcome.Insensitive,
                        $"{option} {Describe(fixtureValue)} -> {Describe(candidate)} reached the rebuilt model, yet parameter count, "
                        + "parameters and outputs are bit-identical to the control both before and after the fixed training steps. "
                        + "Read the option where the model is built or trained, or, if it genuinely cannot act at test scale, mark "
                        + "it [OptionBehaviourExempt(\"reason\")].");
            }
            finally
            {
                (perturbed as IDisposable)?.Dispose();
            }
        }

        return new OptionSensitivityReport(modelKey, option, OptionSensitivityOutcome.NoValidPerturbation,
            rejected.Count == 0
                ? $"No generic perturbation exists for {option} of type {property.PropertyType.Name}."
                : $"Every perturbation of {option} (currently {Describe(fixtureValue)}) was rejected: {string.Join("; ", rejected)}.");
    }

    /// <summary>The runtime baseline key of a model type: its generic type definition's full name.</summary>
    public static string ModelKey(Type modelType)
    {
        var definition = modelType.IsGenericType ? modelType.GetGenericTypeDefinition() : modelType;
        return definition.FullName ?? definition.Name;
    }

    /// <summary>
    /// The options type that configures a model: the first public constructor parameter, scanning
    /// constructors from fewest parameters, whose type derives from <see cref="ModelOptions"/>. The
    /// same rule as ADNOPT001 and the scaffold generator.
    /// </summary>
    public static Type? FindDeclaredOptionsType(Type modelType)
    {
        foreach (var constructor in modelType.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
                     .OrderBy(c => c.GetParameters().Length))
        {
            foreach (var parameter in constructor.GetParameters())
            {
                if (typeof(ModelOptions).IsAssignableFrom(parameter.ParameterType))
                    return parameter.ParameterType;
            }
        }

        return null;
    }

    /// <summary>A content fingerprint over a sequence of values, sensitive to every bit and to length.</summary>
    public static long Fingerprint(IEnumerable<double> values, long seed = unchecked((long)0xcbf29ce484222325UL))
    {
        long hash = seed;
        long count = 0;
        foreach (double value in values)
        {
            hash = unchecked((hash ^ BitConverter.DoubleToInt64Bits(value)) * (long)0x100000001b3UL);
            count++;
        }

        return unchecked((hash ^ count) * (long)0x100000001b3UL);
    }

    /// <summary>Combines fingerprints in order.</summary>
    public static long Combine(params long[] parts)
    {
        long hash = unchecked((long)0xcbf29ce484222325UL);
        foreach (long part in parts)
            hash = unchecked((hash ^ part) * (long)0x100000001b3UL);
        return unchecked((hash ^ parts.Length) * (long)0x100000001b3UL);
    }

    private sealed class ControlFingerprint
    {
        public long Untrained { get; set; }
        public long? Trained { get; set; }
        public string? Failure { get; set; }
        public OptionSensitivityOutcome FailureOutcome { get; set; }
    }

    private static ControlFingerprint MeasureControl<TModel>(OptionSensitivityProbe<TModel> probe, Type declaredOptions)
        where TModel : class
    {
        var result = new ControlFingerprint();
        TModel first;
        TModel second;
        try
        {
            first = Rebuild(probe, declaredOptions, mutate: null);
        }
        catch (Exception ex) when (IsConstructionRejection(ex))
        {
            result.Failure = $"The unperturbed model could not be rebuilt through its clone plan ({Unwrap(ex).GetType().Name}: "
                + $"{Unwrap(ex).Message}), so no perturbed copy can reach it either.";
            result.FailureOutcome = OptionSensitivityOutcome.OptionsNotRouted;
            return result;
        }

        try
        {
            second = Rebuild(probe, declaredOptions, mutate: null);
        }
        catch
        {
            (first as IDisposable)?.Dispose();
            throw;
        }

        try
        {
            long a = probe.Observe(first);
            long b = probe.Observe(second);
            if (a != b)
            {
                result.Failure = "Two identical, identically seeded builds already differ before training, so no difference can be "
                    + "attributed to an option.";
                result.FailureOutcome = OptionSensitivityOutcome.NonDeterministic;
                return result;
            }

            result.Untrained = a;
            if (probe.Train is null) return result;

            try
            {
                probe.Train(first);
                probe.Train(second);
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                // The fixture's model cannot take the generic training step; only the untrained
                // comparison is available, and an option equal there is reported as insensitive with
                // that limitation stated.
                return result;
            }

            long trainedA = probe.Observe(first);
            long trainedB = probe.Observe(second);
            if (trainedA != trainedB)
            {
                result.Failure = "Two identical, identically seeded builds diverge during the fixed training steps, so a "
                    + "training-time difference cannot be attributed to an option.";
                result.FailureOutcome = OptionSensitivityOutcome.NonDeterministic;
                return result;
            }

            result.Trained = trainedA;
            return result;
        }
        finally
        {
            (first as IDisposable)?.Dispose();
            (second as IDisposable)?.Dispose();
        }
    }

    /// <summary>
    /// Builds the fixture's model, swaps its options for an independent (optionally perturbed) copy,
    /// and reconstructs it through the clone plan under the fixed initialisation seed.
    /// </summary>
    private static TModel Rebuild<TModel>(OptionSensitivityProbe<TModel> probe, Type declaredOptions, Action<ModelOptions>? mutate)
        where TModel : class
    {
        var shell = Seeded(probe.Create);
        try
        {
            var held = FindHeldOptions(shell, declaredOptions)
                ?? throw new InvalidOperationException($"The model holds no {declaredOptions.Name} instance.");
            var copy = (ModelOptions)CloneEngine.CopyConfiguration(held);
            probe.PrepareOptions?.Invoke(copy);
            PinSeed(copy);
            mutate?.Invoke(copy);
            ReplaceReferences(shell, held, copy);
            return Seeded(() => (TModel)CloneEngine.CopyConfiguration(shell));
        }
        finally
        {
            (shell as IDisposable)?.Dispose();
        }
    }

    /// <summary>Pins a null <c>Seed</c> so both builds draw any option-seeded randomness identically.</summary>
    private static void PinSeed(ModelOptions options)
    {
        if (options.Seed is null) options.Seed = FixedSeed;
    }

    private static TResult Seeded<TResult>(Func<TResult> build)
    {
        int? previousFallback = LayerInitializationSeedScope.AmbientFallbackSeed;
        var previousScope = LayerInitializationSeedScope.CaptureScope();
        try
        {
            LayerInitializationSeedScope.AmbientFallbackSeed = FixedSeed;
            LayerInitializationSeedScope.ResetForModelConstruction(FixedSeed);
            return build();
        }
        finally
        {
            LayerInitializationSeedScope.AmbientFallbackSeed = previousFallback;
            LayerInitializationSeedScope.RestoreScope(previousScope);
        }
    }

    private static IEnumerable<FieldInfo> InstanceFields(Type type)
    {
        for (var walk = type; walk is not null && walk != typeof(object); walk = walk.BaseType)
        {
            foreach (var field in walk.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
                yield return field;
        }
    }

    /// <summary>The options instance a model holds: the first instance field, most-derived first, of the declared type.</summary>
    private static object? FindHeldOptions(object model, Type declaredOptions)
    {
        foreach (var field in InstanceFields(model.GetType()))
        {
            if (!field.FieldType.IsAssignableFrom(declaredOptions) && !declaredOptions.IsAssignableFrom(field.FieldType)) continue;
            var value = field.GetValue(model);
            if (value is not null && declaredOptions.IsInstanceOfType(value)) return value;
        }

        return null;
    }

    /// <summary>Points every field of the model that holds <paramref name="original"/> at <paramref name="replacement"/>.</summary>
    private static void ReplaceReferences(object model, object original, object replacement)
    {
        foreach (var field in InstanceFields(model.GetType()))
        {
            if (field.FieldType.IsValueType || !field.FieldType.IsInstanceOfType(replacement)) continue;
            if (ReferenceEquals(field.GetValue(model), original)) field.SetValue(model, replacement);
        }
    }

    private static PropertyInfo? FindWritableProperty(Type optionsType, string name)
    {
        // Most-derived declaration first, so a `new` re-declaration wins over the hidden base one.
        for (var walk = optionsType; walk is not null; walk = walk.BaseType)
        {
            var property = walk.GetProperty(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.DeclaredOnly);
            if (property is not null)
                return property.CanWrite && property.SetMethod is { IsPublic: true } ? property : null;
        }

        return null;
    }

    /// <summary>Candidate valid different values, most preferred first.</summary>
    internal static IEnumerable<object?> Perturbations(Type propertyType, object? current)
    {
        var underlying = Nullable.GetUnderlyingType(propertyType) ?? propertyType;

        if (underlying == typeof(bool))
        {
            yield return current is bool flag ? !flag : true;
            yield break;
        }

        if (underlying.IsEnum)
        {
            var values = Enum.GetValues(underlying).Cast<object>().Distinct().ToList();
            int index = current is null ? -1 : values.FindIndex(v => v.Equals(current));
            for (int step = 1; step <= values.Count; step++)
            {
                var candidate = values[((index < 0 ? 0 : index) + step) % values.Count];
                if (!candidate.Equals(current)) yield return candidate;
            }

            yield break;
        }

        if (IsNumeric(underlying))
        {
            foreach (var candidate in NumericCandidates(underlying, current))
                yield return candidate;
            yield break;
        }

        var element = ListElementType(propertyType);
        if (element is not null && current is IEnumerable sequence)
        {
            var items = sequence.Cast<object?>().ToList();
            var grown = new List<object?>(items);
            grown.Add(items.Count > 0 ? items[items.Count - 1] : DefaultElement(element));
            yield return BuildList(propertyType, element, grown);
            if (items.Count > 1)
                yield return BuildList(propertyType, element, items.Take(items.Count - 1).ToList());
        }
    }

    private static IEnumerable<object> NumericCandidates(Type type, object? current)
    {
        double value = current is null ? 0.0 : Convert.ToDouble(current, System.Globalization.CultureInfo.InvariantCulture);
        bool integral = type != typeof(double) && type != typeof(float) && type != typeof(decimal);
        var raw = current is null
            ? new[] { 1.0, 2.0 }
            : integral
                ? new[] { value * 2, value + 1, value - 1 }
                : new[] { value * 2, value / 2, value + 1, value - 1 };

        var seen = new HashSet<double>();
        foreach (double candidate in raw)
        {
            if (candidate == value && current is not null) continue;
            if (double.IsNaN(candidate) || double.IsInfinity(candidate) || !seen.Add(candidate)) continue;
            object? converted;
            try
            {
                converted = Convert.ChangeType(candidate, type, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (OverflowException)
            {
                continue;
            }

            if (converted is not null) yield return converted;
        }
    }

    private static bool IsNumeric(Type type)
        => type == typeof(byte) || type == typeof(sbyte) || type == typeof(short) || type == typeof(ushort)
        || type == typeof(int) || type == typeof(uint) || type == typeof(long) || type == typeof(ulong)
        || type == typeof(float) || type == typeof(double) || type == typeof(decimal);

    private static Type? ListElementType(Type type)
    {
        if (type.IsArray && type.GetArrayRank() == 1) return type.GetElementType();
        if (type.IsGenericType && type.GetGenericArguments().Length == 1
            && typeof(IEnumerable).IsAssignableFrom(type) && type != typeof(string))
        {
            return type.GetGenericArguments()[0];
        }

        return null;
    }

    private static object? DefaultElement(Type element)
    {
        var underlying = Nullable.GetUnderlyingType(element) ?? element;
        if (IsNumeric(underlying)) return Convert.ChangeType(1, underlying, System.Globalization.CultureInfo.InvariantCulture);
        if (underlying == typeof(bool)) return true;
        if (underlying.IsEnum) return Enum.GetValues(underlying).Cast<object>().FirstOrDefault();
        return underlying.IsValueType ? Activator.CreateInstance(underlying) : null;
    }

    private static object BuildList(Type propertyType, Type element, List<object?> items)
    {
        var array = Array.CreateInstance(element, items.Count);
        for (int i = 0; i < items.Count; i++) array.SetValue(items[i], i);
        if (propertyType.IsArray) return array;

        var listType = typeof(List<>).MakeGenericType(element);
        var list = (IList)(Activator.CreateInstance(listType)
            ?? throw new InvalidOperationException($"Cannot create {listType.Name}."));
        foreach (var item in items) list.Add(item);
        if (propertyType.IsInstanceOfType(list)) return list;

        throw new NotSupportedException($"A {propertyType.Name} cannot be rebuilt from a list of {element.Name}.");
    }

    private static bool ValuesEqual(object? left, object? right)
    {
        if (left is null || right is null) return left is null && right is null;
        if (left is IEnumerable a && right is IEnumerable b && left is not string)
            return a.Cast<object?>().SequenceEqual(b.Cast<object?>());
        return left.Equals(right);
    }

    private static string Describe(object? value)
    {
        if (value is null) return "null";
        if (value is string text) return "\"" + text + "\"";
        if (value is IEnumerable sequence)
            return "[" + string.Join(", ", sequence.Cast<object?>().Select(Describe)) + "]";
        return Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture) ?? value.GetType().Name;
    }

    private static Exception Unwrap(Exception ex)
        => ex is TargetInvocationException { InnerException: { } inner } ? Unwrap(inner) : ex;

    /// <summary>
    /// Whether an exception thrown while rebuilding is the constructor refusing the configuration,
    /// which moves on to the next candidate, rather than a harness fault, which propagates.
    /// </summary>
    private static bool IsConstructionRejection(Exception ex)
    {
        var inner = Unwrap(ex);
        return inner is ArgumentException or InvalidOperationException or NotSupportedException
            or NotImplementedException or IndexOutOfRangeException or DivideByZeroException or OverflowException
            or AiDotNet.Exceptions.TensorShapeMismatchException;
    }
}
