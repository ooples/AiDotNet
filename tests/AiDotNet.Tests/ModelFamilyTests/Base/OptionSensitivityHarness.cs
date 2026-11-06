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
    /// <param name="describe">
    /// The observation broken into named parts (the output, each layer's parameters), in a fixed order. Used only to
    /// name what differs when two identical builds disagree, so a NonDeterministic verdict points at its source.
    /// </param>
    /// <param name="prepareModel">
    /// The fixture's post-construction wiring, applied to every rebuilt model: setup the factory performs on the
    /// instance after its constructor (a graph model's synthetic adjacency) is not part of the clone plan, so the
    /// rebuild would otherwise drop it.
    /// </param>
    /// <param name="parameterCount">Optional parameter count measured before fitting/loading fixture assets.</param>
    /// <param name="parameterShapes">Optional ordered parameter tensor shapes before loading fixture assets.</param>
    /// <param name="controlGroup">Configuration context used to keep prepared control measurements separate.</param>
    public OptionSensitivityProbe(
        Func<TModel> create,
        Func<TModel, long> observe,
        Action<TModel>? train,
        Action<ModelOptions>? prepareOptions = null,
        Func<TModel, IReadOnlyList<KeyValuePair<string, long>>>? describe = null,
        Action<TModel>? prepareModel = null,
        Func<TModel, long>? parameterCount = null,
        Func<TModel, IReadOnlyList<int[]>>? parameterShapes = null,
        string? controlGroup = null)
    {
        Create = create ?? throw new ArgumentNullException(nameof(create));
        Observe = observe ?? throw new ArgumentNullException(nameof(observe));
        Train = train;
        PrepareOptions = prepareOptions;
        Describe = describe;
        PrepareModel = prepareModel;
        ParameterCount = parameterCount;
        ParameterShapes = parameterShapes;
        ControlGroup = controlGroup ?? string.Empty;
    }

    /// <summary>The fixture's factory.</summary>
    public Func<TModel> Create { get; }

    /// <summary>The observation fingerprint.</summary>
    public Func<TModel, long> Observe { get; }

    /// <summary>The fixed training steps, or null when the family cannot train generically.</summary>
    public Action<TModel>? Train { get; }

    /// <summary>The options preparation shared by control and perturbed builds.</summary>
    public Action<ModelOptions>? PrepareOptions { get; }

    /// <summary>The observation as named parts, or null when the family does not break it down.</summary>
    public Func<TModel, IReadOnlyList<KeyValuePair<string, long>>>? Describe { get; }

    /// <summary>The fixture's post-construction wiring, re-applied to each rebuilt model.</summary>
    public Action<TModel>? PrepareModel { get; }

    /// <summary>Parameter count before the fixture loads fitted assets, or null for families without parameters.</summary>
    public Func<TModel, long>? ParameterCount { get; }

    /// <summary>Ordered parameter tensor shapes before the fixture loads fitted assets.</summary>
    public Func<TModel, IReadOnlyList<int[]>>? ParameterShapes { get; }

    /// <summary>The prepared configuration context of the cached control.</summary>
    public string ControlGroup { get; }



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
/// <b>Perturbations.</b> A valid different value of the same type: small integral steps before doubling,
/// floating-point values doubled, halved or shifted; a flag flipped; an enum moved to its next member; a list lengthened by
/// repeating its last element, else shortened. Candidates are tried until one changes observable
/// behaviour. Insensitivity is reported only after every accepted candidate has been measured.
/// </para>
/// </remarks>
public static class OptionSensitivityHarness
{
    /// <summary>The layer-initialisation seed both builds share.</summary>
    public const int FixedSeed = 1337;

    /// <summary>The most enum values one option is perturbed to; matches the largest numeric candidate set.</summary>
    internal const int MaxEnumCandidates = 6;

    private static readonly ConcurrentDictionary<(Type Fixture, string Group), Lazy<ControlFingerprint>> s_controls = new();

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
        OptionSensitivityReport report;
        try
        {
            report = Measure(fixtureType, option, probe);
        }
        catch (Exception ex)
        {
            PaperFidelityResultLog.Record(PaperFidelityCheck.OptionSensitivity, fixtureType, null, option,
                PaperFidelityResultLog.ErrorOutcome, $"{Unwrap(ex).GetType().Name}: {Unwrap(ex).Message}");
            throw;
        }

        PaperFidelityResultLog.Record(PaperFidelityCheck.OptionSensitivity, fixtureType, report.ModelKey, report.Option,
            report.Outcome.ToString(), report.Outcome == OptionSensitivityOutcome.Sensitive ? null : report.Detail);
        Enforce(report);
    }

    /// <summary>
    /// Whether an outcome may be listed in <see cref="RuntimeFidelityBaseline.OptionSensitivity"/>.
    /// <see cref="OptionSensitivityOutcome.NonDeterministic"/> and <see cref="OptionSensitivityOutcome.UnknownOption"/>
    /// are not: the first means the harness could not attribute a difference, the second that the generator
    /// and the runtime disagree about the options type. Both are harness or model bugs to fix, never known
    /// defects to carry, so they always fail.
    /// </summary>
    public static bool IsBaselineMaterial(OptionSensitivityOutcome outcome) =>
        outcome != OptionSensitivityOutcome.Sensitive
        && outcome != OptionSensitivityOutcome.NonDeterministic
        && outcome != OptionSensitivityOutcome.UnknownOption;

    /// <summary>Applies the runtime baseline to a measurement: pass, skip as known, or fail.</summary>
    public static void Enforce(OptionSensitivityReport report)
    {
        if (report is null) throw new ArgumentNullException(nameof(report));
        bool baselined = RuntimeFidelityBaseline.OptionSensitivity.TryGetValue(report.BaselineKey, out var expected);
        Assert.False(baselined && !IsBaselineMaterial(expected),
            $"'{report.BaselineKey}' is listed in RuntimeFidelityBaseline.OptionSensitivity as {expected}, which is a harness "
            + "or model bug rather than a known defect; remove the entry and fix the cause.");
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
            var prepared = (ModelOptions)CloneEngine.CopyConfiguration(held);
            probe.PrepareOptions?.Invoke(prepared);
            fixtureValue = property?.GetValue(prepared);
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

        var control = s_controls.GetOrAdd((fixtureType, probe.ControlGroup),
            _ => new Lazy<ControlFingerprint>(() => MeasureControl(probe, declaredOptions))).Value;
        if (control.Failure is not null)
        {
            return new OptionSensitivityReport(modelKey, option, control.FailureOutcome, control.Failure);
        }

        OptionSensitivityReport? insensitive = null;
        var rejected = new List<string>();
        var notHeld = new List<string>();
        var measured = new List<string>();
        foreach (var candidate in Perturbations(property.PropertyType, fixtureValue, option))
        {
            TModel? perturbed;
            // What the options property reads back after the set: a setter may clamp or normalise the candidate.
            object? applied = candidate;
            try
            {
                perturbed = Rebuild(probe, declaredOptions, options =>
                {
                    property.SetValue(options, candidate);
                    applied = property.GetValue(options);
                }, expectedParameterCount: control.ParameterCount,
                    expectedParameterShapesFingerprint: control.ParameterShapesFingerprint);
            }
            catch (ParameterCountChangedException change)
            {
                return new OptionSensitivityReport(modelKey, option, OptionSensitivityOutcome.Sensitive,
                    $"{option} {Describe(fixtureValue)} -> {Describe(candidate)} changes the parameter count from {change.Previous} to {change.Current} before fixture fitting or execution.");
            }
            catch (ParameterShapesChangedException)
            {
                return new OptionSensitivityReport(modelKey, option, OptionSensitivityOutcome.Sensitive,
                    $"{option} {Describe(fixtureValue)} -> {Describe(candidate)} changes the ordered parameter tensor shapes before fixture fitting or execution.");
            }
            catch (Exception ex) when (IsConstructionRejection(ex))
            {
                rejected.Add($"{Describe(candidate)} ({Unwrap(ex).GetType().Name}: {Unwrap(ex).Message})");
                continue;
            }

            try
            {
                if (ValuesEqual(applied, fixtureValue))
                {
                    rejected.Add($"{Describe(candidate)} (the options setter keeps {Describe(applied)})");
                    continue;
                }

                var routed = FindHeldOptions(perturbed, declaredOptions);
                if (routed is null)
                {
                    return new OptionSensitivityReport(modelKey, option, OptionSensitivityOutcome.OptionsNotRouted,
                        $"The rebuilt model holds no {declaredOptions.Name}: its clone plan does not pass the options "
                        + "instance back into the constructor, so no perturbation can reach it.");
                }

                // A model that resets an out-of-range candidate to the value it already had has not shown that the
                // option is unrouted, only that this candidate is invalid for it; another candidate decides. A held
                // value that differs from both is the model's own normalisation of the candidate, which is routed.
                var held = property.GetValue(routed);
                if (!ValuesEqual(held, applied) && ValuesEqual(held, fixtureValue))
                {
                    notHeld.Add(Describe(candidate));
                    continue;
                }

                measured.Add(Describe(candidate));

                long untrained;
                try
                {
                    untrained = Seeded(() => probe.Observe(perturbed));
                }
                catch (Exception ex) when (IsInputContractRejection(ex))
                {
                    return new OptionSensitivityReport(modelKey, option, OptionSensitivityOutcome.Sensitive,
                        $"with {option} = {Describe(candidate)} the model rejects the fixture input "
                        + $"({Unwrap(ex).GetType().Name}), so the option changes its "
                        + (Unwrap(ex) is AiDotNet.TextToSpeech.Latest.KaniSpeechSpanException ? "generated speech contract." : "input contract."));
                }

                if (untrained != control.Untrained)
                {
                    return new OptionSensitivityReport(modelKey, option, OptionSensitivityOutcome.Sensitive,
                        $"{option} {Describe(fixtureValue)} -> {Describe(candidate)} changes parameters or outputs before training.");
                }

                if (probe.Train is null || control.Trained is null)
                {
                    insensitive = new OptionSensitivityReport(modelKey, option, OptionSensitivityOutcome.Insensitive,
                        $"{option} {Describe(fixtureValue)} -> {{{string.Join(", ", measured)}}} reached the rebuilt model and "
                        + "changed neither its parameters nor its outputs; "
                        + (control.TrainingFailure is null
                            ? "this family has no generic training step to look further."
                            : $"the probe's training step failed on the unperturbed model ({control.TrainingFailure}), so "
                              + "training-time effects were not observed."));
                    continue;
                }

                long trained;
                try
                {
                    Seeded(() => probe.Train(perturbed));
                    trained = Seeded(() => probe.Observe(perturbed));
                }
                catch (Exception ex) when (IsInputContractRejection(ex))
                {
                    return new OptionSensitivityReport(modelKey, option, OptionSensitivityOutcome.Sensitive,
                        $"with {option} = {Describe(candidate)} training rejects the fixture data where the control trains "
                        + $"({Unwrap(ex).GetType().Name}: {Unwrap(ex).Message}).");
                }

                if (trained != control.Trained.Value)
                    return new OptionSensitivityReport(modelKey, option, OptionSensitivityOutcome.Sensitive,
                        $"{option} {Describe(fixtureValue)} -> {Describe(candidate)} changes parameters or outputs after training.");
                insensitive = new OptionSensitivityReport(modelKey, option, OptionSensitivityOutcome.Insensitive,
                        $"{option} {Describe(fixtureValue)} -> {{{string.Join(", ", measured)}}} reached the rebuilt model, yet parameter count, "
                        + "parameters and outputs are bit-identical to the control both before and after the fixed training steps. "
                        + "Read the option where the model is built or trained, or, if it genuinely cannot act at test scale, mark "
                        + "it [OptionBehaviourExempt(\"reason\")].");
            }
            finally
            {
                (perturbed as IDisposable)?.Dispose();
            }
        }

        if (insensitive is not null) return insensitive;
        if (notHeld.Count > 0)
        {
            return new OptionSensitivityReport(modelKey, option, OptionSensitivityOutcome.OptionsNotRouted,
                $"The rebuilt model kept {option} = {Describe(fixtureValue)} for every candidate ({string.Join(", ", notHeld)}): "
                + "its clone plan does not pass the options instance back into the constructor, or the model overwrites "
                + "the option, so no perturbation can reach it.");
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
        public long? ParameterCount { get; set; }
        public long? ParameterShapesFingerprint { get; set; }
        public long Untrained { get; set; }
        public long? Trained { get; set; }
        public string? Failure { get; set; }
        public OptionSensitivityOutcome FailureOutcome { get; set; }
        /// <summary>Why the control could not take the probe's training step, when it could not.</summary>
        public string? TrainingFailure { get; set; }
    }

    private static ControlFingerprint MeasureControl<TModel>(OptionSensitivityProbe<TModel> probe, Type declaredOptions)
        where TModel : class
    {
        var result = new ControlFingerprint();
        TModel first;
        TModel second;
        long? secondParameterCount = null;
        long? secondParameterShapesFingerprint = null;
        try
        {
            first = Rebuild(probe, declaredOptions, mutate: null, recordParameterCount: count => result.ParameterCount = count,
                recordParameterShapesFingerprint: hash => result.ParameterShapesFingerprint = hash);
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
            second = Rebuild(probe, declaredOptions, mutate: null, recordParameterCount: count => secondParameterCount = count,
                recordParameterShapesFingerprint: hash => secondParameterShapesFingerprint = hash);
        }
        catch
        {
            (first as IDisposable)?.Dispose();
            throw;
        }

        try
        {
            long a = Seeded(() => probe.Observe(first));
            long b = Seeded(() => probe.Observe(second));
            if (a != b || result.ParameterCount != secondParameterCount || result.ParameterShapesFingerprint != secondParameterShapesFingerprint)
            {
                result.Failure = "Two identical, identically seeded builds already differ before training, so no difference can be "
                    + "attributed to an option." + DescribeDifference(probe, first, second);
                result.FailureOutcome = OptionSensitivityOutcome.NonDeterministic;
                return result;
            }

            result.Untrained = a;
            if (probe.Train is null) return result;

            try
            {
                Seeded(() => probe.Train(first));
                Seeded(() => probe.Train(second));
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                // The fixture's model cannot take the generic training step; only the untrained
                // comparison is available, and an option equal there is reported as insensitive with
                // that limitation -- and the exception that imposed it -- stated.
                result.TrainingFailure = $"{Unwrap(ex).GetType().Name}: {Unwrap(ex).Message}";
                return result;
            }

            long trainedA = Seeded(() => probe.Observe(first));
            long trainedB = Seeded(() => probe.Observe(second));
            if (trainedA != trainedB)
            {
                result.Failure = "Two identical, identically seeded builds diverge during the fixed training steps, so a "
                    + "training-time difference cannot be attributed to an option." + DescribeDifference(probe, first, second);
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
    /// Names the parts of the observation on which two builds disagree (at most five), or an empty string when the
    /// probe has no breakdown or the breakdown itself throws.
    /// </summary>
    private static string DescribeDifference<TModel>(OptionSensitivityProbe<TModel> probe, TModel first, TModel second)
        where TModel : class
    {
        if (probe.Describe is null) return string.Empty;
        try
        {
            var a = probe.Describe(first);
            var b = probe.Describe(second);
            if (a.Count != b.Count)
                return $" The builds have different structures ({a.Count} against {b.Count} observed parts).";
            var differing = new List<string>();
            for (int i = 0; i < a.Count; i++)
                if (a[i].Value != b[i].Value) differing.Add(a[i].Key);
            if (differing.Count == 0) return string.Empty;
            string shown = string.Join(", ", differing.Take(5));
            string more = differing.Count > 5 ? $" and {differing.Count - 5} more" : string.Empty;
            return $" Differing: {shown}{more} ({differing.Count} of {a.Count} parts).";
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            return string.Empty;
        }
    }

    private sealed class ParameterShapesChangedException : Exception { }

    private static IEnumerable<double> ParameterShapeValues(IReadOnlyList<int[]> shapes)
    {
        foreach (var shape in shapes)
        {
            yield return shape.Length;
            foreach (int dimension in shape) yield return dimension;
        }
    }

    private sealed class ParameterCountChangedException : Exception
    {
        internal ParameterCountChangedException(long previous, long current)
        {
            Previous = previous;
            Current = current;
        }
        internal long Previous { get; }
        internal long Current { get; }
    }

    /// <summary>
    /// Builds the fixture's model, swaps its options for an independent (optionally perturbed) copy,
    /// and reconstructs it through the clone plan under the fixed initialisation seed.
    /// </summary>
    private static TModel Rebuild<TModel>(OptionSensitivityProbe<TModel> probe, Type declaredOptions, Action<ModelOptions>? mutate,
        long? expectedParameterCount = null, Action<long>? recordParameterCount = null,
        long? expectedParameterShapesFingerprint = null, Action<long>? recordParameterShapesFingerprint = null)
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
            var rebuilt = Seeded(() => (TModel)CloneEngine.CopyConfiguration(shell));
            try
            {
                if (probe.ParameterCount is not null)
                {
                    long count = Seeded(() => probe.ParameterCount(rebuilt));
                    recordParameterCount?.Invoke(count);
                    if (expectedParameterCount.HasValue && expectedParameterCount.Value != count)
                        throw new ParameterCountChangedException(expectedParameterCount.Value, count);
                }
                if (probe.ParameterShapes is not null)
                {
                    long hash = Seeded(() => Fingerprint(ParameterShapeValues(probe.ParameterShapes(rebuilt))));
                    recordParameterShapesFingerprint?.Invoke(hash);
                    if (expectedParameterShapesFingerprint.HasValue && expectedParameterShapesFingerprint.Value != hash)
                        throw new ParameterShapesChangedException();
                }
                Seeded(() => probe.PrepareModel?.Invoke(rebuilt));
                return rebuilt;
            }
            catch
            {
                (rebuilt as IDisposable)?.Dispose();
                throw;
            }
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

    // Lazy parameters and component RNGs may first be created while observing or training.
    // Keep those operations under the same fixture seed as construction.
    private static void Seeded(Action action)
    {
        Seeded(() => { action(); return 0; });
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
    internal static IEnumerable<object?> Perturbations(Type propertyType, object? current, string? option = null)
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
            // Every candidate is a rebuild plus training, and an option that changes nothing is only concluded after
            // all of them: TimeSeriesModelType's ~70 values ran Timer past the 300 s budget. Like the numeric
            // candidates, at most MaxEnumCandidates, spread evenly over the enum from the next value so a small enum
            // is still covered completely.
            // An unset value has every member as a candidate; a set one has all the others.
            int others = index < 0 ? values.Count : values.Count - 1;
            int count = Math.Min(MaxEnumCandidates, others);
            for (int i = 0; i < count; i++)
            {
                int step = 1 + (int)((long)i * others / count);
                var candidate = values[(index + step) % values.Count];
                if (!candidate.Equals(current)) yield return candidate;
            }

            yield break;
        }

        if (IsNumeric(underlying))
        {
            IEnumerable<object> candidates = NumericCandidates(underlying, current);
            if (option is "Beta1" or "Beta2" or "AdamBeta1" or "AdamBeta2")
                candidates = candidates.Where(value => Convert.ToDouble(value) >= 0 && Convert.ToDouble(value) < 1);
            if (option is "TeacherMomentumStart" or "TeacherMomentumEnd" || option?.EndsWith("Probability", StringComparison.Ordinal) == true)
                candidates = candidates.Where(value => Convert.ToDouble(value) >= 0 && Convert.ToDouble(value) <= 1);
            if (option == "TopP")
                candidates = candidates.Where(value => Convert.ToDouble(value) > 0 && Convert.ToDouble(value) <= 1);
            if (option == "FilterThreshold")
                candidates = candidates.Where(value => Convert.ToDouble(value) >= 0 && Convert.ToDouble(value) < 1);
            if (option == "RepetitionPenalty")
                candidates = candidates.Where(value => Convert.ToDouble(value) > 0 && Convert.ToDouble(value) < 2);
            if (option == "NumMels" && current is int)
                candidates = candidates.Concat(new object[] { 80, 128 }).Where(value => !Equals(value, current)).Distinct();
            if (option?.EndsWith("HeadDim", StringComparison.Ordinal) == true
                && current is int headDim && headDim > 0 && headDim <= int.MaxValue / 2)
                // Rotary embeddings need even head widths: prefer a valid structural change before +/-1.
                candidates = new object[] { headDim * 2 }.Concat(candidates).Distinct();
            if (option == "SampleRate" && current is int rate && rate > 0)
            {
                // A one-Hz change can make a polyphase resampler's ratio coprime and allocate gigabytes.
                // Simple ratios test the same setting first; retain the fine-grained candidates as fallbacks.
                var preferred = new List<object>();
                if (rate <= int.MaxValue / 2) preferred.Add(rate * 2);
                if (rate > 1 && rate % 2 == 0) preferred.Add(rate / 2);
                candidates = preferred.Concat(candidates).Distinct();
            }
            foreach (var candidate in candidates)
                yield return candidate;
            yield break;
        }

        var element = ListElementType(propertyType);
        if (element is not null && current is null)
        {
            // An unset list (null usually means "none") is perturbed to one default element.
            yield return BuildList(propertyType, element, new List<object?> { DefaultElement(element) });
            yield break;
        }

        if (element is not null && current is IEnumerable sequence)
        {
            var items = sequence.Cast<object?>().ToList();
            if (option?.EndsWith("UpsampleRates", StringComparison.Ordinal) == true && element == typeof(int) && items.Count > 1)
            {
                // Equal rates cannot be reordered. Transfer a factor between two stages so the
                // product (and the coupled waveform hop) stays fixed while stage geometry changes.
                for (int from = 0; from < items.Count; from++)
                    for (int to = 0; to < items.Count; to++)
                        if (from != to && items[from] is int source && items[to] is int destination
                            && source > 1 && source % 2 == 0 && destination > 0 && destination <= int.MaxValue / 2)
                        {
                            var redistributed = new List<object?>(items);
                            redistributed[from] = source / 2;
                            redistributed[to] = destination * 2;
                            yield return BuildList(propertyType, element, redistributed);
                        }
            }
            // Stage arrays often require an exact length. Change one scalar element at a time
            // as well as the length, without sharing mutable containers with the fixture.
            var scalar = Nullable.GetUnderlyingType(element) ?? element;
            if (IsNumeric(scalar) || scalar == typeof(bool) || scalar.IsEnum)
            {
                // Reordering stage rates preserves their product and length, unlike changing one rate alone.
                var reversed = items.AsEnumerable().Reverse().ToList();
                if (items.Count > 1 && !items.SequenceEqual(reversed))
                    yield return BuildList(propertyType, element, reversed);

                for (int index = 0; index < items.Count; index++)
                {
                    foreach (var value in Perturbations(element, items[index]))
                    {
                        var changed = new List<object?>(items);
                        changed[index] = value;
                        yield return BuildList(propertyType, element, changed);
                    }
                }
            }
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
        // Prefer a one-step structural change before potentially exponential depth/width expansions.
        var raw = current is null
            ? new[] { 1.0, 2.0 }
            : integral
                ? new[] { value + 1, value - 1, value + 2, value - 2, value / 2, 1.0, value * 2 }
                : new[] { value * 2, value / 2, value + 0.1, value - 0.1, value + 1, value + 2, 1.0, value - 1 };

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
    /// <summary>
    /// True when a perturbed model refuses the fixture's data the way a changed input contract does: an argument or
    /// shape rejection, or an explicit "not supported". Only that is evidence the option changed behaviour. Any other
    /// failure (a null dereference, an engine or tape fault) is a defect, so it propagates and the case is recorded
    /// as an Error rather than passing as Sensitive.
    /// </summary>
    private static bool IsInputContractRejection(Exception ex)
    {
        var inner = Unwrap(ex);
        return inner is ArgumentException or NotSupportedException or AiDotNet.Exceptions.TensorShapeMismatchException
            or AiDotNet.TextToSpeech.Latest.KaniSpeechSpanException;
    }

    private static bool IsConstructionRejection(Exception ex)
    {
        var inner = Unwrap(ex);
        return inner is ArgumentException or InvalidOperationException or NotSupportedException
            or NotImplementedException or IndexOutOfRangeException or DivideByZeroException or OverflowException
            or AiDotNet.Exceptions.TensorShapeMismatchException;
    }
}
