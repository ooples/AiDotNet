using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Clustering.Options;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Configuration;

/// <summary>
/// Options classes must not hide an inherited property with <c>new</c>.
/// </summary>
/// <remarks>
/// <para>
/// A hiding property splits one setting into two storage slots. Code that holds the options as the derived type
/// reads one, code that holds them as the base type (an optimizer base clamping its learning rate, a clustering
/// base choosing its metric, cloning and AutoML setting <c>Seed</c>) reads the other, and a value set through
/// either is invisible to the other. Sixty such properties existed; each was either turned into an override of a
/// virtual base property or deleted with its default moved into the derived constructor.
/// </para>
/// <para>
/// <see cref="NoOptionsPropertyHidesAnInheritedOne"/> keeps the count at zero.
/// <see cref="InheritedOption_KeepsTheDerivedDefaultAndIsOneSlot"/> checks each repaired class: the default the
/// shadow used to carry is what the BASE-typed property now reports, and a value written through the base is read
/// back through the derived type.
/// </para>
/// </remarks>
public class ShadowedOptionsTests
{
    [Fact(Timeout = 120000)]
    public async Task NoOptionsPropertyHidesAnInheritedOne()
    {
        await Task.Yield();
        var hiding = new List<string>();
        foreach (var type in SafeTypes(typeof(ModelOptions).Assembly).Where(t => t.IsClass && typeof(ModelOptions).IsAssignableFrom(t)))
        {
            foreach (var property in type.GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly))
            {
                var accessor = property.GetGetMethod() ?? property.GetSetMethod();
                if (accessor is null || accessor.GetBaseDefinition().DeclaringType != accessor.DeclaringType) continue;
                for (var ancestor = type.BaseType; ancestor is not null && ancestor != typeof(object); ancestor = ancestor.BaseType)
                {
                    if (ancestor.GetProperty(property.Name, BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly) is not null)
                    {
                        hiding.Add($"{type.Name}.{property.Name} hides {ancestor.Name}.{property.Name}");
                        break;
                    }
                }
            }
        }

        Assert.True(hiding.Count == 0,
            "Options properties hiding an inherited one (override the virtual base property, or delete the shadow and set "
            + "its default in the constructor):\n  " + string.Join("\n  ", hiding));
    }

    public static IEnumerable<object?[]> RepairedOptions()
    {
        // Overrides of the virtual OptimizationAlgorithmOptions learning-rate bounds.
        yield return Case<OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>>(new AdaMaxOptimizerOptions<double, Matrix<double>, Vector<double>>(), "MinLearningRate", 1e-5, 2e-5);
        yield return Case<OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>>(new AdaMaxOptimizerOptions<double, Matrix<double>, Vector<double>>(), "MaxLearningRate", 0.1, 0.2);
        yield return Case<OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>>(new AMSGradOptimizerOptions<double, Matrix<double>, Vector<double>>(), "MinLearningRate", 1e-5, 2e-5);
        yield return Case<OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>>(new AMSGradOptimizerOptions<double, Matrix<double>, Vector<double>>(), "MaxLearningRate", 0.1, 0.2);
        yield return Case<OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>>(new BFGSOptimizerOptions<double, Matrix<double>, Vector<double>>(), "MinLearningRate", 1e-6, 2e-6);
        yield return Case<OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>>(new BFGSOptimizerOptions<double, Matrix<double>, Vector<double>>(), "MaxLearningRate", 10.0, 5.0);
        yield return Case<OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>>(new ConjugateGradientOptimizerOptions<double, Matrix<double>, Vector<double>>(), "MinLearningRate", 1e-6, 2e-6);
        yield return Case<OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>>(new ConjugateGradientOptimizerOptions<double, Matrix<double>, Vector<double>>(), "MaxLearningRate", 1.0, 0.5);
        yield return Case<OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>>(new DFPOptimizerOptions<double, Matrix<double>, Vector<double>>(), "MinLearningRate", 1e-6, 2e-6);
        yield return Case<OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>>(new DFPOptimizerOptions<double, Matrix<double>, Vector<double>>(), "MaxLearningRate", 10.0, 5.0);
        yield return Case<OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>>(new FTRLOptimizerOptions<double, Matrix<double>, Vector<double>>(), "MaxLearningRate", 0.1, 0.2);
        yield return Case<OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>>(new LBFGSOptimizerOptions<double, Matrix<double>, Vector<double>>(), "MinLearningRate", 1e-6, 2e-6);
        yield return Case<OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>>(new LBFGSOptimizerOptions<double, Matrix<double>, Vector<double>>(), "MaxLearningRate", 10.0, 5.0);
        yield return Case<OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>>(new MiniBatchGradientDescentOptions<double, Matrix<double>, Vector<double>>(), "MaxLearningRate", 0.1, 0.2);
        yield return Case<OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>>(new MomentumOptimizerOptions<double, Matrix<double>, Vector<double>>(), "MaxLearningRate", 0.1, 0.2);
        yield return Case<OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>>(new NadamOptimizerOptions<double, Matrix<double>, Vector<double>>(), "MinLearningRate", 1e-5, 2e-5);
        yield return Case<OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>>(new NadamOptimizerOptions<double, Matrix<double>, Vector<double>>(), "MaxLearningRate", 0.1, 0.2);

        // Shadows deleted, default moved into the derived constructor.
        yield return Case<OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>>(new MomentumOptimizerOptions<double, Matrix<double>, Vector<double>>(), "MomentumIncreaseFactor", 1.02, 1.03);
        yield return Case<OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>>(new MomentumOptimizerOptions<double, Matrix<double>, Vector<double>>(), "MomentumDecreaseFactor", 0.98, 0.97);
        yield return Case<OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>>(new NesterovAcceleratedGradientOptimizerOptions<double, Matrix<double>, Vector<double>>(), "MomentumIncreaseFactor", 1.02, 1.03);
        yield return Case<OptimizationAlgorithmOptions<double, Matrix<double>, Vector<double>>>(new NesterovAcceleratedGradientOptimizerOptions<double, Matrix<double>, Vector<double>>(), "MomentumDecreaseFactor", 0.98, 0.97);
        yield return Case<GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>>(new CMAESOptimizerOptions<double, Matrix<double>, Vector<double>>(), "PopulationSize", 4 + (int)(3 * Math.Log(100)), 30);
        yield return Case<GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>>(new CMAESOptimizerOptions<double, Matrix<double>, Vector<double>>(), "MaxGenerations", 100, 7);
        yield return Case<GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>>(new DifferentialEvolutionOptions<double, Matrix<double>, Vector<double>>(), "CrossoverRate", 0.5, 0.6);
        yield return Case<GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>>(new DifferentialEvolutionOptions<double, Matrix<double>, Vector<double>>(), "MutationRate", 0.8, 0.7);
        yield return Case<GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>>(new DifferentialEvolutionOptions<double, Matrix<double>, Vector<double>>(), "PopulationSize", 50, 60);
        yield return Case<GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>>(new TabuSearchOptions<double, Matrix<double>, Vector<double>>(), "MutationRate", 0.1, 0.2);
        yield return Case<GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>>(new TabuSearchOptions<double, Matrix<double>, Vector<double>>(), "MinMutationRate", 0.01, 0.02);
        yield return Case<GeneticAlgorithmOptimizerOptions<double, Matrix<double>, Vector<double>>>(new TabuSearchOptions<double, Matrix<double>, Vector<double>>(), "MaxMutationRate", 0.5, 0.4);
        yield return Case<TimeSeriesRegressionOptions<double>>(new SARIMAOptions<double>(), "SeasonalPeriod", 12, 4);
        yield return Case<TimeSeriesRegressionOptions<double>>(new STLDecompositionOptions<double>(), "SeasonalPeriod", 12, 4);
        yield return Case<TimeSeriesRegressionOptions<double>>(new UnobservedComponentsOptions<double, Matrix<double>, Vector<double>>(), "SeasonalPeriod", 1, 4);
        yield return Case<DecisionTreeOptions>(new NGBoostRegressionOptions<double>(), "MinSamplesSplit", 2, 5);
        yield return Case<ReinforcementLearningOptions<double>>(new TD3Options<double>(), "WarmupSteps", 25000, 100);

        // Seed: every repaired class keeps the inherited null default.
        yield return Case<ModelOptions>(new ResidualBootstrapFitDetectorOptions(), "Seed", null, 7);
        yield return Case<ModelOptions>(new FinancialDQNAgentOptions<double>(), "Seed", null, 7);
        yield return Case<ModelOptions>(new AiDotNet.Audio.AudioGen.AudioGenOptions(), "Seed", null, 7);
        yield return Case<ModelOptions>(new AiDotNet.ReinforcementLearning.Policies.BetaPolicyOptions<double>(), "Seed", null, 7);
        yield return Case<ModelOptions>(new AiDotNet.ReinforcementLearning.Policies.ContinuousPolicyOptions<double>(), "Seed", null, 7);
        yield return Case<ModelOptions>(new AiDotNet.ReinforcementLearning.Policies.DeterministicPolicyOptions<double>(), "Seed", null, 7);
        yield return Case<ModelOptions>(new AiDotNet.ReinforcementLearning.Policies.DiscretePolicyOptions<double>(), "Seed", null, 7);
        yield return Case<ModelOptions>(new AiDotNet.ReinforcementLearning.Policies.MixedPolicyOptions<double>(), "Seed", null, 7);
        yield return Case<ModelOptions>(new AiDotNet.ReinforcementLearning.Policies.MultiModalPolicyOptions<double>(), "Seed", null, 7);

        // DistanceMetric: the clustering base reads the inherited property; every repaired class shares it.
        foreach (var options in new ClusteringOptions<double>[]
        {
            new AffinityPropagationOptions<double>(), new BIRCHOptions<double>(), new CLARANSOptions<double>(),
            new DBSCANOptions<double>(), new DenclueOptions<double>(), new FuzzyCMeansOptions<double>(),
            new GMeansOptions<double>(), new HDBSCANOptions<double>(), new HierarchicalOptions<double>(),
            new KMedoidsOptions<double>(), new MeanShiftOptions<double>(), new OPTICSOptions<double>(),
            new COPKMeansOptions<double>(), new SeededKMeansOptions<double>(), new SOMOptions<double>(),
            new SpectralOptions<double>(), new OnlineKMeansOptions<double>(), new XMeansOptions<double>(),
        })
        {
            yield return Case<ClusteringOptions<double>>(options, "DistanceMetric", null, new EuclideanDistance<double>());
        }
    }

    [Theory(Timeout = 120000)]
    [MemberData(nameof(RepairedOptions))]
    public async Task InheritedOption_KeepsTheDerivedDefaultAndIsOneSlot(string name, object options, Type baseType, string property, object? expectedDefault, object written)
    {
        await Task.Yield();
        var inherited = baseType.GetProperty(property, BindingFlags.Public | BindingFlags.Instance)
            ?? throw new InvalidOperationException($"{baseType.Name} declares no {property}.");
        // GetProperty on the runtime type would be ambiguous if a shadow came back; it fails the test then.
        var derived = options.GetType().GetProperty(property, BindingFlags.Public | BindingFlags.Instance)
            ?? throw new InvalidOperationException($"{name} exposes no {property}.");

        Assert.Equal(expectedDefault, inherited.GetValue(options));
        inherited.SetValue(options, written);
        Assert.Equal(written, derived.GetValue(options));
    }

    private static object?[] Case<TBase>(TBase options, string property, object? expectedDefault, object written)
        where TBase : class
        => new object?[] { $"{options.GetType().Name}.{property}", options, typeof(TBase), property, expectedDefault, written };

    private static IEnumerable<Type> SafeTypes(Assembly assembly)
    {
        try
        {
            return assembly.GetTypes();
        }
        catch (ReflectionTypeLoadException ex)
        {
            return ex.Types.Where(t => t is not null).Cast<Type>();
        }
    }
}
