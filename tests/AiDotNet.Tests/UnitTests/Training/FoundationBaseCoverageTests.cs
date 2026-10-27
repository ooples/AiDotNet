using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.UnitTests.Training;

/// <summary>Accounts for all parameterized model bases without constructing large models.</summary>
public class FoundationBaseCoverageTests
{
    private readonly ITestOutputHelper _output;
    public FoundationBaseCoverageTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void EveryParameterizedBase_HasStreamingDispatchOrAnExplicitPackedContract()
    {
        // These contracts own host vectors / tree topology, or shard-gather and cache side effects.
        // Replacing their setters with raw tensor writes would change their semantics.
        var packedRoots = new HashSet<string>
        {
            "RegressionBase", "NonLinearRegressionBase", "DecisionTreeRegressionBase",
            "AsyncDecisionTreeRegressionBase", "ShardedModelBase", "ClassicalVolatilityModelBase"
        };
        var missing = new List<string>();
        int inspected = 0;
        foreach (var definition in typeof(AiDotNet.Models.ModelBase<,,>).Assembly.GetTypes()
            .Where(t => t.IsAbstract && t.IsClass && t.Name.Split('`')[0].EndsWith("Base", StringComparison.Ordinal))
            .OrderBy(t => t.FullName))
        {
            if (!definition.GetInterfaces().Any(i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(IParameterizable<,,>))) continue;
            var type = definition;
            if (type.ContainsGenericParameters)
            {
                var count = type.GetGenericArguments().Length;
                Assert.True(count == 1 || count == 3, $"Unclassified generic base {type}");
                type = type.MakeGenericType(count == 1 ? new[] { typeof(double) }
                    : new[] { typeof(double), typeof(Matrix<double>), typeof(Vector<double>) });
            }
            var contract = type.GetInterfaces().First(i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(IParameterizable<,,>));
            var map = type.GetInterfaceMap(contract);
            bool packed = false;
            for (Type? parent = type; parent is not null; parent = parent.BaseType)
                packed |= packedRoots.Contains(parent.Name.Split('`')[0]);
            for (int i = 0; i < map.InterfaceMethods.Length; i++)
            {
                if (map.InterfaceMethods[i].Name is not ("GetParameterChunks" or "SetParameterChunks")) continue;
                var target = map.TargetMethods[i];
                _output.WriteLine($"{definition.FullName}: {target.Name} -> {target.DeclaringType}");
                if (target.DeclaringType?.IsInterface == true && !packed) missing.Add($"{definition.FullName}.{target.Name}");
            }
            inspected++;
        }
        Assert.True(inspected >= 30, $"Unexpectedly small base inventory: {inspected}");
        Assert.True(missing.Count == 0, "Unclassified flat fallback: " + string.Join(", ", missing));
        _output.WriteLine($"Inspected {inspected} parameterized base classes.");
    }
}
