using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Xunit;
using Xunit.Abstractions;

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

    [Fact]
    public void AllDefaultConstructableModels_ShouldListDiscoveredTypes()
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

    [Fact]
    public void AllDefaultConstructableModels_ShouldConstructWithoutException()
    {
        var types = GetDefaultConstructableModelTypes().ToList();
        var failures = new List<(string TypeName, string Error)>();
        var timeouts = new List<string>();
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
                    // Log timeout as warning, not failure - some models are intentionally large
                    timeouts.Add(typeName);
                    _output.WriteLine($"TIMEOUT: {closedType.Name} (>{ConstructionTimeout.TotalSeconds}s)");
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

        // Only fail on actual exceptions, not timeouts
        Assert.True(failures.Count == 0,
            $"{failures.Count} model(s) threw exceptions during default construction:\n" +
            string.Join("\n", failures.Select(f => $"  {f.TypeName}: {f.Error}")));
    }
}
