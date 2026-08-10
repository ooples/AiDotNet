using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Xunit;
using Xunit.Abstractions;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests;

/// <summary>
/// Integration tests that verify all model types can be default-constructed without crashing.
/// This catches NullReferenceException, ArgumentNullException, ArgumentOutOfRangeException,
/// and construction hangs that were discovered by the CrowdTrainer project (Issue #915).
/// </summary>
public class DefaultConstructionTests
{
    private readonly ITestOutputHelper _output;

    /// <summary>
    /// Maximum time to allow for a single model construction before treating it as a hang.
    /// </summary>
    private static readonly TimeSpan ConstructionTimeout = TimeSpan.FromSeconds(10);

    /// <summary>
    /// How long to let a timed-out construction actually finish before moving on, so its CPU cost
    /// is not charged to the next model measured.
    /// </summary>
    private static readonly TimeSpan StragglerDrainTimeout = TimeSpan.FromSeconds(30);

    /// <summary>Total draining allowed across the whole sweep, not per straggler.</summary>
    /// <remarks>
    /// PER-TYPE DRAINING COULD EXCEED THE TEST'S OWN TIMEOUT. The sweep runs under
    /// <c>[Fact(Timeout = 120000)]</c>. At 10 s per construction plus 30 s of draining, one
    /// straggler cost up to 40 s and three consumed the entire budget, at which point xUnit aborts
    /// the test -- which reports NOTHING, while the abandoned constructor threads keep running into
    /// the next test anyway. That is strictly worse than the CPU-charging problem draining exists to
    /// solve.
    ///
    /// So the budget is shared across the sweep and checked before each drain. Once it is spent,
    /// later stragglers are recorded as undrained and the sweep keeps moving, which is the same
    /// outcome the per-type bound produced for a genuine deadlock.
    /// </remarks>
    private static readonly TimeSpan TotalStragglerDrainBudget = TimeSpan.FromSeconds(45);

    public DefaultConstructionTests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// Gets all concrete generic types that have a parameterless constructor or
    /// all-optional-parameter constructor, from the AiDotNet assembly.
    /// </summary>
    private static IEnumerable<Type> GetDefaultConstructableModelTypes()
    {
        var assembly = typeof(AiDotNet.Models.ModelMetadata<>).Assembly;
        var types = assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition)
            .Where(t => t.GetGenericArguments().Length == 1);

        foreach (var openType in types)
        {
            Type closedType;
            try
            {
                closedType = openType.MakeGenericType(typeof(double));
            }
            catch
            {
                continue;
            }

            // Check if IFullModel is implemented
            var implementsFullModel = closedType.GetInterfaces()
                .Any(i => i.IsGenericType &&
                          i.GetGenericTypeDefinition().Name.StartsWith("IFullModel"));

            if (!implementsFullModel)
                continue;

            // Check for parameterless or all-optional constructor
            var constructors = closedType.GetConstructors(BindingFlags.Public | BindingFlags.Instance);
            var hasDefaultCtor = constructors.Any(c =>
            {
                var parameters = c.GetParameters();
                return parameters.Length == 0 || parameters.All(p => p.HasDefaultValue);
            });

            if (hasDefaultCtor)
                yield return closedType;
        }
    }

    [Fact(Timeout = 120000)]
    public async Task AllDefaultConstructableModels_ShouldListDiscoveredTypes()
    {
        var types = GetDefaultConstructableModelTypes().ToList();
        _output.WriteLine($"Found {types.Count} default-constructable IFullModel types:");
        foreach (var type in types.OrderBy(t => t.FullName))
        {
            _output.WriteLine($"  {type.FullName}");
        }

        Assert.True(types.Count > 50,
            $"Expected at least 50 default-constructable IFullModel types, found {types.Count}. " +
            "This may indicate a regression in default constructor availability.");
    }

    [Fact(Timeout = 120000)]
    public async Task AllDefaultConstructableModels_ShouldConstructWithoutException()
    {
        var types = GetDefaultConstructableModelTypes().ToList();
        var failures = new List<(string TypeName, string Error)>();
        var timeouts = new List<string>();
        var undrained = new List<string>();
        var successes = 0;
        var drainSpent = new System.Diagnostics.Stopwatch();

        foreach (var closedType in types)
        {
            var typeName = closedType.FullName ?? closedType.Name;
            try
            {
                var ctor = closedType.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
                    .Where(c => c.GetParameters().Length == 0 ||
                                c.GetParameters().All(p => p.HasDefaultValue))
                    .OrderBy(c => c.GetParameters().Length)
                    .First();

                var parameters = ctor.GetParameters()
                    .Select(p => p.DefaultValue == DBNull.Value ? null : p.DefaultValue)
                    .ToArray();

                // Use a timeout to catch hangs
                var task = System.Threading.Tasks.Task.Run(() => ctor.Invoke(parameters));
                if (!task.Wait(ConstructionTimeout))
                {
                    timeouts.Add(typeName);
                    _output.WriteLine($"TIMEOUT: {closedType.Name} (>{ConstructionTimeout.TotalSeconds}s)");

                    // Drain the straggler before measuring the next model. Task.Wait() only stops
                    // WAITING — the construction keeps running on its thread pool thread. Without
                    // this, every timed-out model's CPU cost is charged to its innocent successors,
                    // so one genuinely slow constructor cascades into a string of spurious timeouts
                    // further down the list. That made this test order- and load-dependent: CI
                    // reported DreamerAgent, a busy dev machine reported VideoCLIP instead, and
                    // "fix the named model, watch a different one appear" looked like whack-a-mole
                    // when it was one leak.
                    //
                    // Bounded, so a genuine deadlock cannot hang the suite; anything still running
                    // after the drain is counted and reported rather than silently accumulating.
                    var remainingDrain = TotalStragglerDrainBudget - drainSpent.Elapsed;
                    if (remainingDrain <= TimeSpan.Zero)
                    {
                        // The shared budget is gone. Recording rather than draining keeps the sweep
                        // inside its own timeout; an aborted test reports nothing at all.
                        undrained.Add(typeName);
                        continue;
                    }

                    var thisDrain = remainingDrain < StragglerDrainTimeout ? remainingDrain : StragglerDrainTimeout;
                    drainSpent.Start();
                    bool drained = task.Wait(thisDrain);
                    drainSpent.Stop();

                    if (!drained)
                    {
                        undrained.Add(typeName);
                    }
                    continue;
                }

                if (task.Exception is not null)
                {
                    var innerEx = task.Exception.InnerException ?? task.Exception;
                    failures.Add((typeName,
                        $"{innerEx.GetType().Name}: {innerEx.Message}"));
                    continue;
                }

                successes++;
            }
            catch (TargetInvocationException tie) when (tie.InnerException is not null)
            {
                failures.Add((typeName,
                    $"{tie.InnerException.GetType().Name}: {tie.InnerException.Message}"));
            }
            catch (Exception ex)
            {
                failures.Add((typeName,
                    $"{ex.GetType().Name}: {ex.Message}"));
            }
        }

        _output.WriteLine($"\nResults: {successes} OK, {failures.Count} FAILED, {timeouts.Count} TIMEOUT out of {types.Count} total");

        if (failures.Count > 0)
        {
            _output.WriteLine("\nFailures (these are bugs that must be fixed):");
            foreach (var (tn, error) in failures)
            {
                _output.WriteLine($"  FAIL: {tn} - {error}");
            }
        }

        if (timeouts.Count > 0)
        {
            _output.WriteLine("\nTimeouts (construction took too long, likely creating large parameter arrays):");
            foreach (var tn in timeouts)
            {
                _output.WriteLine($"  TIMEOUT: {tn}");
            }
        }

        if (undrained.Count > 0)
        {
            _output.WriteLine(
                $"\n{undrained.Count} construction(s) were still running after the " +
                $"{StragglerDrainTimeout.TotalSeconds}s drain — a genuine deadlock, or a constructor " +
                "far slower than the gate. Measurements taken after these are less reliable:");
            foreach (var tn in undrained)
            {
                _output.WriteLine($"  UNDRAINED: {tn}");
            }
        }

        // Gate on genuine constructor EXCEPTIONS only, never on timeouts. This comment always said
        // exactly that, but timeouts were being added to `failures` as well, so they gated the shard
        // regardless — the code did the opposite of what it documented.
        //
        // A timeout here is a statement about how loaded the machine was, not about whether the
        // model can be constructed: this same sweep named DreamerAgent on CI and VideoCLIP on a busy
        // dev box, from identical source. Reporting them is useful; failing the build on them turns
        // machine load into a red shard and sends people to fix models that were never broken.
        Assert.True(failures.Count == 0,
            $"{failures.Count} model(s) threw exceptions during default construction:\n" +
            string.Join("\n", failures.Select(f => $"  {f.TypeName}: {f.Error}")));

        // AND A FLOOR ON WHAT WAS ACTUALLY VERIFIED. Gating on failures.Count alone meant timeouts
        // and undrained stragglers were printed and discarded, so a change that made every
        // constructor slow -- or a heavily loaded runner -- drove every type into the timeout
        // branch, left failures empty, and reported green having constructed nothing.
        //
        // The floor is a fraction rather than a fixed count so it does not need editing as models
        // are added, and it is deliberately loose: an individual timeout still must not fail the
        // shard (that is the whole point of the gate above), but a systemic regression turns it red.
        int minimumSuccesses = Math.Max(1, (types.Count * 2) / 3);
        Assert.True(successes >= minimumSuccesses,
            $"Only {successes} of {types.Count} models constructed successfully, below the floor of " +
            $"{minimumSuccesses}. {timeouts.Count} timed out and {undrained.Count} were still running " +
            "after the shared drain budget. A sweep that verifies almost nothing must not report " +
            "green: this is a systemic regression or a runner problem, not an individual slow model.");
    }
}
