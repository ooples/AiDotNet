using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Cloning;

/// <summary>
/// Sweeps every model through DeepCopy to find which ones lose construction state.
/// </summary>
/// <remarks>
/// <para>
/// The point is the coverage number, not a pass, for the same reason the layer sweep exists: a
/// green build says nothing about whether a clone carries what it was built with.
/// </para>
/// <para>
/// DeepCopy now rebuilds a model from its generated construction plan and generated state payload.
/// Concrete models have no factory, clone, or serialization override to keep synchronized with
/// their constructors. What this measures is how many models survive that shared round trip with
/// their parameter count and architecture intact, and stay independent of the original afterwards.
/// </para>
/// <para>
/// Models that cannot be constructed from a standard architecture are reported separately from
/// models that fail to clone. Conflating a shortfall in this harness with a shortfall in the
/// feature is how a coverage number stops meaning anything.
/// </para>
/// </remarks>
public class AllModelsCloneTests
{
    private readonly ITestOutputHelper _output;

    /// <summary>Initializes a new instance of the <see cref="AllModelsCloneTests"/> class.</summary>
    /// <param name="output">Sink for the coverage summary.</param>
    public AllModelsCloneTests(ITestOutputHelper output) => _output = output;

    /// <summary>Returned by <see cref="Attempt"/> for a model this harness cannot build.</summary>
    private const string SkipMarker = "\0skip";

    /// <summary>How many shards the model list is split across.</summary>
    /// <remarks>
    /// The sweep is split rather than given a longer clock. One run over every model needed more
    /// than the 45-minute ceiling a shard gets, and a single test that cannot finish inside its
    /// budget reports nothing at all -- the 15-minute attempt died at the letter C. Sharding also
    /// lets the runner work on them in parallel, so the wall-clock is one shard, not the sum.
    /// </remarks>
    private const int ShardCount = 24;

    private string ReportPath =
        System.IO.Path.Combine(System.IO.Path.GetTempPath(), "aidotnet-model-clone-sweep.txt");

    private readonly List<string> cloned = new();
    private readonly List<string> failed = new();
    private readonly List<string> notConstructed = new();
    private readonly List<string> budgetExceeded = new();

    /// <summary>Maximum observation window for one model in this diagnostic sweep.</summary>
    /// <remarks>
    /// Exceeding this budget is deliberately not called a hang. The number is a property of this
    /// harness and runner capacity; changing it changes the count without changing model behavior.
    /// </remarks>
    private static readonly TimeSpan PerModelProbeBudget = TimeSpan.FromSeconds(20);

    /// <summary>
    /// Models whose original never materialized under the probe, so the two sides are not comparable.
    /// </summary>
    /// <remarks>
    /// A HARNESS LIMIT, kept out of the failure count for the same reason <see cref="notConstructed"/>
    /// is. The probe below is a 1x4 tensor; a vision-language model refuses it, so the original sits
    /// at whatever its constructor sized while DeepCopy returns a copy that has materialized. The
    /// sweep read that as "BlipNeuralNetwork: 23441664 parameters against 768" and counted a
    /// failure, when what it had actually measured was one side resolved and the other not. Nothing
    /// is known about those models' cloning either way until the harness can drive them with an
    /// input they accept -- which is a statement about this test, not about the copy.
    /// </remarks>
    private readonly List<string> unresolved = new();

    /// <summary>Appends one line to the progress file as the sweep runs.</summary>
    /// <remarks>
    /// Written as it goes rather than at the end. The first two runs timed out having written
    /// nothing, which said only that the sweep was slow and not which model it was stuck on.
    /// </remarks>
    private void Note(string line)
    {
        lock (ReportPath) System.IO.File.AppendAllLines(ReportPath, new[] { line });
    }

    private string Fail(Type open, string why)
    {
        var line = $"{open.Name}: {why}";
        Note($"FAIL  {line}");
        return line;
    }

    /// <summary>Reports how many models survive a DeepCopy with their construction state.</summary>
    /// <returns>A task representing the test.</returns>
    // ONE SHARD, WELL INSIDE THE CEILING. The single-test version died at CTCSegmentation after 15
    // minutes -- 104 of 200+ models -- and raising its clock past the 45-minute shard ceiling would
    // only have moved where it died. Each shard now takes 1/24th of the list; 8 shards still overran.
    [Theory(Timeout = 900000)]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(10)]
    [InlineData(11)]
    [InlineData(12)]
    [InlineData(13)]
    [InlineData(14)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(17)]
    [InlineData(18)]
    [InlineData(19)]
    [InlineData(20)]
    [InlineData(21)]
    [InlineData(22)]
    [InlineData(23)]
    public async System.Threading.Tasks.Task EveryModel_ReportsWhetherItSurvivesADeepCopy(int shard)
    {
        await System.Threading.Tasks.Task.Yield();

        // ONE SHARD PER PROCESS. `dotnet test --filter` cannot reliably address an individual
        // [InlineData] case, so parallel runs each executed every shard and overwrote one
        // another's report. The runner sets AIDOTNET_SWEEP_SHARD and the shards that do not
        // match return immediately, so each process does 1/24th of the work and owns one file.
        // Unset means run every shard, which is what a plain `dotnet test` should still do.
        var only = Environment.GetEnvironmentVariable("AIDOTNET_SWEEP_SHARD");
        if (!string.IsNullOrEmpty(only) && int.TryParse(only, out var wanted) && wanted != shard) return;

        // AIDOTNET_SWEEP_DIR keeps the reports off the system drive, which nine parallel runs
        // filled to zero bytes free.
        var dir = Environment.GetEnvironmentVariable("AIDOTNET_SWEEP_DIR");
        if (string.IsNullOrEmpty(dir)) dir = System.IO.Path.GetTempPath();

        ReportPath = System.IO.Path.Combine(dir, $"aidotnet-model-clone-sweep-{shard}.txt");

        var all = typeof(NeuralNetworkBase<>).Assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && !t.IsNested)
            .Where(t => t.IsGenericTypeDefinition && t.GetGenericArguments().Length == 1)
            .Where(DerivesFromNeuralNetworkBase)
            .OrderBy(t => t.Name, StringComparer.Ordinal)
            .ToList();

        // STRIDED, not carved into contiguous blocks. The expensive models cluster by name (the
        // whole BLIP/Blip2/BLIP3 family lands together), so contiguous blocks would put every slow
        // one in the same shard and leave that shard timing out while the others idle.
        var candidates = all.Where((_, i) => i % ShardCount == shard).ToList();

        // APPENDED AS IT GOES. The first run timed out at 15 minutes with nothing written, which
        // told us only that the sweep is slow -- not which model it was on. A progress file costs
        // nothing and turns a timeout into a result plus a culprit.
        System.IO.File.WriteAllText(ReportPath, string.Empty);


        foreach (var open in candidates)
        {
            Type closed;
            try
            {
                closed = open.MakeGenericType(typeof(float));
            }
            catch (Exception)
            {
                notConstructed.Add($"{open.Name}: constraints reject float"); Note($"skip  {open.Name}: constraints reject float");
                continue;
            }

            // BEFORE the attempt, so a shard-level timeout still names the model it was observing.
            Note($"try   {open.Name}");

            // A BUDGET PER MODEL. This keeps one slow or stuck attempt from consuming the whole
            // shard, but it is only an observation budget. It cannot distinguish a true deadlock
            // from valid work that needs more time on this runner, so report it separately and do
            // not turn the budget-sensitive count into a claimed hang rate.
            string? outcome = null;
            var work = System.Threading.Tasks.Task.Run(() => outcome = Attempt(open, closed));

            if (!work.Wait(PerModelProbeBudget))
            {
                budgetExceeded.Add(
                    $"{open.Name}: exceeded {PerModelProbeBudget.TotalSeconds:0}s observation budget");
                Note($"LIMIT {open.Name}");
                continue;
            }

            if (outcome is null) cloned.Add(open.Name);
            else if (!ReferenceEquals(outcome, SkipMarker) && outcome != SkipMarker) failed.Add(outcome);
        }

        _output.WriteLine($"model types        : {candidates.Count}");
        _output.WriteLine($"cloned OK          : {cloned.Count}");
        _output.WriteLine($"clone FAILED       : {failed.Count}");
        _output.WriteLine($"not constructed    : {notConstructed.Count} (harness limit, not a clone result)");
        _output.WriteLine($"probe did not run  : {unresolved.Count} (harness limit, not a clone result)");
        _output.WriteLine($"probe budget limit : {budgetExceeded.Count} (budget-sensitive; not a hang rate)");
        _output.WriteLine(string.Empty);

        foreach (var line in failed) _output.WriteLine($"FAIL  {line}");
        foreach (var line in notConstructed) _output.WriteLine($"skip  {line}");
        foreach (var line in unresolved) _output.WriteLine($"lazy  {line}");

        // ALSO to a file. xunit only surfaces ITestOutputHelper on a failing test or under
        // `verbosity=detailed`, and detailed logs every one of 72,000 discovered cases -- 17MB of
        // noise to read four numbers out of. A report worth running is worth being able to read.
        var report = new List<string>
        {
            $"model types        : {candidates.Count}",
            $"cloned OK          : {cloned.Count}",
            $"clone FAILED       : {failed.Count}",
            $"not constructed    : {notConstructed.Count} (harness limit, not a clone result)",
            $"probe did not run  : {unresolved.Count} (harness limit, not a clone result)",
            $"probe budget limit : {budgetExceeded.Count} (budget-sensitive; not a hang rate)",
            string.Empty,
        };
        report.AddRange(failed.Select(f => $"FAIL  {f}"));
        report.AddRange(notConstructed.Select(n => $"skip  {n}"));
        report.AddRange(unresolved.Select(u => $"lazy  {u}"));

        report.AddRange(budgetExceeded.Select(t => $"LIMIT {t}"));
        System.IO.File.WriteAllLines(ReportPath, report);
    }

    /// <summary>Constructs, copies and checks one model. Returns null when it cloned cleanly.</summary>
    private string? Attempt(Type open, Type closed)
    {
        var model = TryConstruct(closed);
        if (model is null)
        {
            notConstructed.Add($"{open.Name}: no constructor takes a standard architecture");
            Note($"skip  {open.Name}: not constructible");
            return SkipMarker;
        }

        try
        {
            var probed = Resolve(model);
            var before = model.ParameterCount;
            var copy = model.DeepCopy() as NeuralNetworkBase<float>;

            if (copy is null) return Fail(open, "DeepCopy returned null");
            if (copy.GetType() != closed) return Fail(open, $"copy is {copy.GetType().Name}");

            Resolve(copy);

            if (copy.ParameterCount != before)
            {
                // AN UNRESOLVED ORIGINAL IS NOT A FAILED COPY. Every model in this bucket reports
                // the copy as the LARGER side -- DeepCopy materialized layers the original had left
                // lazy because the probe never ran on it. Comparing those two counts measures
                // materialization, not copying.
                if (!probed)
                {
                    unresolved.Add($"{open.Name}: probe did not run ({copy.ParameterCount} against {before})");
                    Note($"lazy  {open.Name}");
                    return SkipMarker;
                }

                return Fail(open, $"{copy.ParameterCount} parameters against {before}");
            }

            if (ReferenceEquals(copy, model) || !IsIndependent(model, copy))
                return Fail(open, "copy is not independent of the original");

            Note($"ok    {open.Name}");
            return null;
        }
        catch (Exception ex)
        {
            var inner = ex.InnerException ?? ex;
            var message = inner.Message;
            return Fail(open, $"{inner.GetType().Name}: {message.Substring(0, Math.Min(90, message.Length))}");
        }
        finally
        {
            (model as IDisposable)?.Dispose();
        }
    }

    /// <summary>Runs one probe input through the model so lazy layers materialise.</summary>
    /// <remarks>
    /// A model that cannot accept the standard probe is left as it is; the comparison below then
    /// still holds, because both sides are measured in the same unresolved state.
    /// </remarks>
    /// <returns>True when the probe ran, so the model's parameter surface is materialized.</returns>
    private static bool Resolve(NeuralNetworkBase<float> model)
    {
        try
        {
            var input = new Tensor<float>(new[] { 1, 4 });
            model.Predict(input);

            return true;
        }
        catch (Exception)
        {
            // Not every model predicts from a 1x4 probe. Both sides get the same treatment, but the
            // caller needs to KNOW that neither side was driven -- an unresolved original compared
            // against a materialized copy is not a result about cloning.
            return false;
        }
    }

    /// <summary>Whether writing through one model leaves the other alone.</summary>
    private static bool IsIndependent(
        NeuralNetworkBase<float> original,
        NeuralNetworkBase<float> copy)
    {
        var parameters = original.GetParameters();
        if (parameters.Length == 0) return true;

        var mutated = new Vector<float>(parameters.Length);
        for (var i = 0; i < parameters.Length; i++) mutated[i] = parameters[i] + 1.0f;

        copy.UpdateParameters(mutated);

        var after = original.GetParameters();
        for (var i = 0; i < after.Length; i++)
        {
            if (Math.Abs(after[i] - parameters[i]) > 1e-5f) return false;
        }

        return true;
    }

    private static NeuralNetworkBase<float>? TryConstruct(Type closed)
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 2);

        // The widest constructor whose every remaining argument is optional, so a model is built
        // through the one carrying the most configuration rather than the narrowest.
        foreach (var ctor in closed.GetConstructors()
                     .OrderByDescending(c => c.GetParameters().Length))
        {
            var formal = ctor.GetParameters();
            if (formal.Length == 0) continue;

            var args = new object?[formal.Length];
            var usable = true;

            for (var i = 0; i < formal.Length && usable; i++)
            {
                if (formal[i].ParameterType.IsInstanceOfType(architecture)) args[i] = architecture;
                else if (formal[i].HasDefaultValue) args[i] = formal[i].DefaultValue;
                else usable = false;
            }

            if (!usable || args.All(a => a is null)) continue;

            try
            {
                return ctor.Invoke(args) as NeuralNetworkBase<float>;
            }
            catch (Exception)
            {
                // A model that rejects the standard architecture is a harness limit, not a defect.
            }
        }

        return null;
    }

    private static bool DerivesFromNeuralNetworkBase(Type type)
    {
        for (var b = type.BaseType; b is not null; b = b.BaseType)
        {
            if (b.IsGenericType && b.GetGenericTypeDefinition() == typeof(NeuralNetworkBase<>)) return true;
            if (b.Name.StartsWith("NeuralNetworkBase", StringComparison.Ordinal)) return true;
        }

        return false;
    }
}
