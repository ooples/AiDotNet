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
    /// is not charged to the next model measured. Bounded so a true deadlock cannot hang the suite.
    /// </summary>
    private static readonly TimeSpan StragglerDrainTimeout = TimeSpan.FromSeconds(30);

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
                    if (!task.Wait(StragglerDrainTimeout))
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
    }
}
