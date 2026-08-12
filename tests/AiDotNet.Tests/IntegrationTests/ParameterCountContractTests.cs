using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests;

/// <summary>
/// Forces the parameter sweeps to run sequentially. xUnit parallelises test CLASSES by default,
/// and each of these constructs every model in the library; run together they doubled peak memory
/// and the host was killed mid-run. Sharing a collection is xUnit's way of saying "not at the
/// same time as each other".
/// </summary>
[CollectionDefinition("ParameterSweeps", DisableParallelization = true)]
public class ParameterSweepCollection { }

/// <summary>
/// Asserts, across EVERY constructable model, that <c>ParameterCount</c> and
/// <c>GetParameters().Length</c> describe the same tensors.
/// </summary>
/// <remarks>
/// <para>
/// This invariant already existed, but only ran for models that have a generated test scaffold. A
/// sweep over every constructable model found 56 violations where CI reported 19 — so roughly two
/// thirds of the breakage was real, live, and invisible, and each fix to a visible one revealed
/// nothing about the rest. That is the shape of a failure list you cannot plan against.
/// </para>
/// <para>
/// The consequence is not cosmetic. 284 call sites slice a flat parameter vector by
/// <c>ParameterCount</c> — <c>SpanBasedNERBase.UpdateParameters</c> is one — so when the two
/// disagree, <c>SetParameters</c> pairs the wrong lengths and a saved checkpoint restores into the
/// wrong tensors. The model then silently keeps its initial weights, which no other test detects.
/// </para>
/// <para>
/// One test over all models rather than one test per model, and it reports EVERY violation in a
/// single message. Failing at the first offender would recreate exactly the problem this exists to
/// solve: you would fix one, re-run, discover the next, and never know how many are left.
/// </para>
/// </remarks>
[Collection("ParameterSweeps")]
[Trait("Category", "Sweep")]
public class ParameterCountContractTests
{
    private readonly ITestOutputHelper _output;

    public ParameterCountContractTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Above this parameter count the sweep reads <c>ParameterCount</c> only and never calls
    /// <c>GetParameters()</c>.
    /// </summary>
    /// <remarks>
    /// Materialising a flat vector costs 8 bytes per parameter for <c>double</c>, on top of the
    /// weights themselves. CogVideoModel's default constructor builds the paper-scale 5B variant
    /// (baseChannels 384, cross-attention dim 4096, 49 frames at 480x720); asking it for a flat
    /// copy is a ~40 GB allocation that killed the test host outright and took the rest of the
    /// sweep with it. The MODEL is fine -- construction is cheap and ParameterCount is just a sum
    /// over its two sub-networks. The sweep was the problem.
    ///
    /// So the guard is on size, not on a list of names: the cheap value decides whether the
    /// expensive one is safe to ask for. 50M parameters is ~400 MB at double, comfortably
    /// measurable, and above it the pairing cannot be checked -- which is reported honestly rather
    /// than skipped silently.
    /// </remarks>
    private const long MaxParametersToMaterialize = 50_000_000L;

    /// <summary>Bounded so one pathological constructor cannot stall the sweep.</summary>
    private static readonly TimeSpan ConstructionTimeout = TimeSpan.FromSeconds(10);

    /// <summary>
    /// Number of disjoint slices the sweep is split into. Every model is measured by exactly one.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The sweep measures ~2,000 models at roughly 4.3 s each — construction plus a full
    /// <c>GetParameters()</c> materialization — and runs three isolated workers at a time, so it
    /// needs about 2.4 h. Against a 30-minute per-test budget it could never do anything but time
    /// out, and a timed-out sweep asserts nothing at all.
    /// </para>
    /// <para>
    /// The cost is the measuring itself, not the harness around it: launching one worker process
    /// per model was measured at ~160 ms, about 2 minutes across the whole sweep — 1.4% — so
    /// batching models into shared processes was not the answer. Nothing here is made cheaper.
    /// Instead the SAME total work is split into <see cref="ShardCount"/> cases, each of which fits
    /// the existing budget with room to spare, and which CI can place in separate matrix jobs.
    /// Coverage is unchanged: every model is still measured, and no assertion is relaxed.
    /// </para>
    /// <para>
    /// Slicing is round-robin (<c>index % ShardCount</c>) rather than contiguous because cost is
    /// heavily clustered by name — the LLaVA, CogVideo and Emu families sit together in discovery
    /// order, and contiguous blocks would hand one shard nearly all of the expensive models while
    /// the rest finished in seconds.
    /// </para>
    /// </remarks>
    public const int ShardCount = 8;

    public static IEnumerable<object[]> Shards =>
        Enumerable.Range(0, ShardCount).Select(i => new object[] { i });

    [Theory(Timeout = 1800000)]
    [MemberData(nameof(Shards))]
    public async System.Threading.Tasks.Task AllModels_ParameterCountMatchesGetParameters(int shardIndex)
    {
        await System.Threading.Tasks.Task.Yield();
#if !NET10_0_OR_GREATER
        return;
#endif

        var violations = new List<string>();
        int checkedCount = 0, skipped = 0, unmeasurable = 0, unsized = 0, constructed = 0;

        // Persist violations as they are found. Constructing ~2,000 models in one process is enough
        // to crash the test host outright -- an AccessViolation or StackOverflow inside a single
        // constructor cannot be caught in-process, and the first attempt at this test died with
        // "Test host process crashed" and produced NOTHING. A gate that can lose its whole result
        // is worse than no gate: it is indistinguishable from a pass. The file survives the crash,
        // and the last line in it names the model that was being measured when the host died.
        // Per shard: the shards of one run would otherwise truncate each other's log, and this file
        // is the only record that survives a test-host crash.
        var logPath = System.IO.Path.Combine(System.IO.Path.GetTempPath(),
            $"parameter-count-violations-shard{shardIndex}of{ShardCount}.txt");
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

        var allModelTypes = GetConstructableModelTypes().ToArray();
        var modelTypes = allModelTypes.Where((_, i) => i % ShardCount == shardIndex).ToArray();
        _output.WriteLine($"Shard {shardIndex} of {ShardCount}: {modelTypes.Length} of " +
                          $"{allModelTypes.Length} constructable models.");

        // A shard that received nothing means the slicing broke, not that the models are healthy.
        Assert.True(modelTypes.Length > 0,
            $"Shard {shardIndex} of {ShardCount} was given no models to measure, so it can only " +
            $"pass vacuously. Discovery returned {allModelTypes.Length} constructable model(s).");

        var measurements = await ParameterSweepProcess.MeasureAllAsync(
            modelTypes, includeChunks: false, MaxParametersToMaterialize, ConstructionTimeout);

        foreach (var result in measurements)
        {
            var closedType = result.ModelType;
            var typeName = closedType.FullName ?? closedType.Name;

            log?.WriteLine($"[measuring] {typeName}");
            constructed++;
            if (constructed % 50 == 0)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }

            var measurement = result.Measurement;
            switch (measurement.Status)
            {
                case "deferred":
                case "unmaterialized":
                    unsized++;
                    log?.WriteLine($"UNSIZED {typeName}: readiness={measurement.Readiness}");
                    continue;
                case "too-large":
                    skipped++;
                    _output.WriteLine($"TOO LARGE TO MEASURE {typeName}: ParameterCount={measurement.Declared}");
                    continue;
                case "unsupported":
                case "no-chunks":
                    skipped++;
                    continue;
                case "ok":
                    break;
                default:
                    unmeasurable++;
                    _output.WriteLine($"UNMEASURABLE {typeName}: {measurement.Status} {measurement.Error}");
                    continue;
            }

            long declared = measurement.Declared;
            long actual = measurement.Flat;
            if (actual < 0) { skipped++; continue; }
            checkedCount++;

            if (declared == 0 && actual == 0 && measurement.Readiness != "ParameterFree")
            {
                var ambiguous = $"{typeName}: both surfaces returned zero, but manifest readiness " +
                                $"was {measurement.Readiness}; zero is valid only for ParameterFree models";
                violations.Add(ambiguous);
                log?.WriteLine("VIOLATION " + ambiguous);
                continue;
            }

            if (declared != actual)
            {
                var v = $"{typeName}: ParameterCount={declared}, " +
                        $"GetParameters().Length={actual} (difference {declared - actual})";
                violations.Add(v);
                log?.WriteLine("VIOLATION " + v);
            }
        }

        _output.WriteLine($"Shard {shardIndex}/{ShardCount}: checked {checkedCount} models; " +
                          $"{skipped} skipped (no flat vector), " +
                          $"{unsized} deferred/unmaterialized; {unmeasurable} unmeasurable; " +
                          $"{violations.Count} violations.");

        Assert.True(checkedCount > 0,
            $"Shard {shardIndex} of {ShardCount} of the isolated parameter-count sweep did not " +
            $"complete a single measurement out of the {modelTypes.Length} models it was given.");

        Assert.True(violations.Count == 0,
            $"Shard {shardIndex} of {ShardCount}: {violations.Count} model(s) report a " +
            "ParameterCount that disagrees with " +
            "GetParameters().Length. The two must describe the same tensors: SetParameters pairs " +
            "them by length, and 284 call sites slice a flat vector by the count, so a mismatch " +
            "means a saved checkpoint restores into the wrong tensors and the model silently keeps " +
            "its initial weights.\n\n" +
            "Usual causes, in the order they have actually occurred here: GetParameters inventing a " +
            "placeholder when the model is empty (fabricating a parameter to satisfy a 'must be " +
            "non-empty' assertion); a count computed from a formula rather than from the tensors " +
            "the getter walks (e.g. states x actions, when not every state has every action); and a " +
            "base that counts derived state the getter excludes (a target network is recomputed by " +
            "SetParameters, not restored by it).\n\n" +
            "The durable fix is to derive one from the other rather than maintain both:\n" +
            "  public override long ParameterCount => GetParameters().Length;\n\n" +
            string.Join("\n", violations.OrderBy(v => v, StringComparer.Ordinal).Select(v => "  " + v)));
    }

    private static long ReadLong(object instance, string propertyName)
    {
        var prop = instance.GetType().GetProperty(propertyName,
            BindingFlags.Public | BindingFlags.Instance | BindingFlags.FlattenHierarchy);
        if (prop is null) return -1;
        try { return Convert.ToInt64(prop.GetValue(instance)); }
        catch { return -1; }
    }

    private static long ReadVectorLength(object instance)
    {
        var m = instance.GetType().GetMethod("GetParameters", BindingFlags.Public | BindingFlags.Instance,
            binder: null, types: Type.EmptyTypes, modifiers: null);
        if (m is null) return -1;
        var vec = m.Invoke(instance, null);
        if (vec is null) return 0;
        var lenProp = vec.GetType().GetProperty("Length");
        return lenProp is null ? -1 : Convert.ToInt64(lenProp.GetValue(vec));
    }

    private static bool TryConstruct(Type closedType, out object? instance)
    {
        instance = null;
        try
        {
            var ctor = closedType.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
                .Where(c => c.GetParameters().Length == 0 ||
                            c.GetParameters().All(p => p.HasDefaultValue))
                .OrderBy(c => c.GetParameters().Length)
                .FirstOrDefault();
            if (ctor is null) return false;

            var args = ctor.GetParameters()
                .Select(p => p.DefaultValue == DBNull.Value ? null : p.DefaultValue)
                .ToArray();

            object? built = null;
            var task = System.Threading.Tasks.Task.Run(() => built = ctor.Invoke(args));
            if (!task.Wait(ConstructionTimeout))
            {
                // Drain rather than abandon. Task.Wait only stops WAITING; the construction keeps
                // running and its CPU cost is then charged to every model measured after it, which
                // is what made DefaultConstructionTests name a different model depending on load.
                task.Wait(TimeSpan.FromSeconds(20));
                return false;
            }
            if (task.Exception is not null) return false;

            instance = built;
            return instance is not null;
        }
        catch
        {
            return false;
        }
    }

    private static IEnumerable<Type> GetConstructableModelTypes()
    {
        var assembly = typeof(AiDotNet.Models.ModelMetadata<>).Assembly;
        var open = assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition)
            .Where(t => t.GetGenericArguments().Length == 1);

        foreach (var openType in open)
        {
            Type closed;
            try { closed = openType.MakeGenericType(typeof(double)); }
            catch { continue; }

            bool isModel = closed.GetInterfaces().Any(i =>
                i.IsGenericType &&
                i.GetGenericTypeDefinition().Name.StartsWith("IFullModel", StringComparison.Ordinal));
            if (!isModel) continue;

            bool constructable = closed.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
                .Any(c => c.GetParameters().Length == 0 || c.GetParameters().All(p => p.HasDefaultValue));
            if (!constructable) continue;

            yield return closed;
        }
    }
}
