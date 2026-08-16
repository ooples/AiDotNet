using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Factories;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Factories;

/// <summary>
/// Guards <see cref="OptimizerFactory{T, TInput, TOutput}"/> against the gap where an
/// <see cref="OptimizerType"/> value names an optimizer this library ships but the factory cannot build it.
/// </summary>
/// <remarks>
/// <para>
/// The factory is the only route from a name to an instance, and both real callers —
/// <c>YamlConfigApplier</c> (which resolves <c>optimizer.type</c> from a YAML config) and
/// <c>TrainerBase</c> — go through it. A registration gap therefore surfaces as a runtime
/// "Unknown optimizer type" for a value the enum openly advertises and the library genuinely implements.
/// </para>
/// <para>
/// These tests are written against the ENUM rather than a hand-maintained list, so adding a new
/// <see cref="OptimizerType"/> without registering its implementation fails here rather than in a user's
/// YAML file.
/// </para>
/// </remarks>
public class OptimizerFactoryTests
{
    /// <summary>
    /// Enum values that name a concept rather than a shipped optimizer. Each is listed with the reason it has
    /// no implementation to resolve to, so this set stays a deliberate exclusion list and not a dumping ground.
    /// </summary>
    public static readonly IReadOnlyDictionary<OptimizerType, string> UnimplementedByDesign =
        new Dictionary<OptimizerType, string>
        {
            [OptimizerType.AdaptiveGradient] = "generic family name; Adagrad is the concrete implementation",
            [OptimizerType.CrossEntropy] = "the cross-entropy optimization METHOD is not implemented",
            [OptimizerType.EvolutionaryAlgorithm] = "family name; GeneticAlgorithm/DifferentialEvolution are concrete",
            [OptimizerType.HillClimbing] = "no HillClimbingOptimizer in this library",
            [OptimizerType.NestedLearning] = "paradigm placeholder; no optimizer class",
            [OptimizerType.QuasiNewton] = "family name; BFGS/DFP are the concrete implementations",
        };

    public static TheoryData<OptimizerType> ImplementedOptimizerTypes
    {
        get
        {
            var data = new TheoryData<OptimizerType>();
            foreach (OptimizerType t in Enum.GetValues(typeof(OptimizerType)))
                if (!UnimplementedByDesign.ContainsKey(t))
                    data.Add(t);
            return data;
        }
    }

    /// <summary>
    /// Every enum value the class declares as having no implementation, so the diagnostic-message test
    /// covers all of them and picks up new entries automatically.
    /// </summary>
    public static TheoryData<OptimizerType> UnimplementedOptimizerTypes
    {
        get
        {
            var data = new TheoryData<OptimizerType>();
            foreach (var t in UnimplementedByDesign.Keys)
                data.Add(t);
            return data;
        }
    }

    /// <summary>
    /// Every enum value that names a shipped optimizer must build with default options.
    /// </summary>
    [Theory]
    [MemberData(nameof(ImplementedOptimizerTypes))]
    public void CreateOptimizer_WithDefaults_BuildsEveryImplementedType(OptimizerType type)
    {
        IOptimizer<double, Matrix<double>, Vector<double>> optimizer;
        try
        {
            optimizer = OptimizerFactory<double, Matrix<double>, Vector<double>>.CreateOptimizer(type);
        }
        catch (Exception ex)
        {
            throw new Xunit.Sdk.XunitException(
                $"OptimizerType.{type} is advertised by the enum but the factory could not build it: " +
                $"{ex.GetType().Name}: {ex.Message}");
        }

        Assert.NotNull(optimizer);

        // Building it is not enough — the instance must be the implementation the enum names, and it must
        // be usable. A factory that quietly fell back to a default optimizer would still produce a
        // non-null object and pass a NotNull-only test.
        //
        // Asserted as a round trip rather than `GetOptimizerType(x) == type`, because several enum values
        // are aliases of one implementation (AdaGrad/Adagrad, Adamax/AdaMax, ...) and the reverse map
        // deliberately reports one canonical name for each. What must hold is that the canonical name
        // rebuilds the same implementation.
        var factoryType = OptimizerFactory<double, Matrix<double>, Vector<double>>.GetOptimizerType(optimizer);
        var rebuilt = OptimizerFactory<double, Matrix<double>, Vector<double>>.CreateOptimizer(factoryType);
        Assert.Equal(optimizer.GetType(), rebuilt.GetType());

        if (optimizer is IGradientBasedOptimizer<double, Matrix<double>, Vector<double>> gradientBased)
        {
            var parameters = new Vector<double>(new[] { 0.5, -0.25, 0.125 });
            var gradient = new Vector<double>(new[] { 0.1, -0.2, 0.3 });
            var updated = gradientBased.UpdateParameters(parameters, gradient);

            Assert.Equal(parameters.Length, updated.Length);
            for (int i = 0; i < updated.Length; i++)
            {
                Assert.False(double.IsNaN(updated[i]),
                    $"OptimizerType.{type} produced NaN at index {i} on its first step from default options.");
                Assert.False(double.IsInfinity(updated[i]),
                    $"OptimizerType.{type} produced an infinity at index {i} on its first step from default options.");
            }
        }
    }

    /// <summary>
    /// The options-taking overload must work too. It is public API, so a caller can reach it directly even
    /// though the in-repo call sites happen to use the defaults overload.
    /// </summary>
    [Fact]
    public void CreateOptimizer_WithOptions_AppliesTheSuppliedOptions()
    {
        var options = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialLearningRate = 0.0123,
            Beta1 = 0.85,
        };

        var optimizer = OptimizerFactory<double, Matrix<double>, Vector<double>>
            .CreateOptimizer(OptimizerType.Adam, options);

        var applied = Assert.IsType<AdamOptimizerOptions<double, Matrix<double>, Vector<double>>>(
            optimizer.GetOptions());
        Assert.Equal(0.0123, applied.InitialLearningRate);
        Assert.Equal(0.85, applied.Beta1);
    }

    /// <summary>
    /// The round trip must be stable: build from a type, ask what type it is, get the same answer back.
    /// </summary>
    /// <remarks>
    /// This is the property alias enum values threaten. <c>Adagrad</c> and <c>AdaGrad</c> both resolve to
    /// <c>AdagradOptimizer</c>, so a reverse lookup that simply scans the forward registry would return
    /// whichever alias the dictionary happened to enumerate first — an ordering-dependent answer, on a method
    /// whose documented purpose is identifying an optimizer for serialization.
    /// </remarks>
    [Theory]
    [MemberData(nameof(ImplementedOptimizerTypes))]
    public void GetOptimizerType_RoundTripsToACanonicalNameForTheSameImplementation(OptimizerType type)
    {
        var optimizer = OptimizerFactory<double, Matrix<double>, Vector<double>>.CreateOptimizer(type);

        var reported = OptimizerFactory<double, Matrix<double>, Vector<double>>.GetOptimizerType(optimizer);

        // Building from the reported name must yield the same implementation type — that is what "canonical"
        // has to mean for serialization to survive a round trip, and it holds for aliases too.
        var rebuilt = OptimizerFactory<double, Matrix<double>, Vector<double>>.CreateOptimizer(reported);
        Assert.Equal(optimizer.GetType(), rebuilt.GetType());
    }

    /// <summary>
    /// Repeated reverse lookups must agree with each other — a dictionary-order-dependent answer would not.
    /// </summary>
    [Fact]
    public void GetOptimizerType_IsDeterministicAcrossCalls()
    {
        var optimizer = new AdagradOptimizer<double, Matrix<double>, Vector<double>>(null);

        var first = OptimizerFactory<double, Matrix<double>, Vector<double>>.GetOptimizerType(optimizer);
        Assert.Equal(OptimizerType.Adagrad, first);
        for (int i = 0; i < 20; i++)
            Assert.Equal(OptimizerType.Adagrad,
                OptimizerFactory<double, Matrix<double>, Vector<double>>.GetOptimizerType(optimizer));
    }

    [Fact]
    public void CreateOptimizer_WithMismatchedOptions_RejectsTheConfiguration()
    {
        var options = new RpropOptimizerOptions<double, Matrix<double>, Vector<double>>();

        var ex = Assert.Throws<ArgumentException>(() =>
            OptimizerFactory<double, Matrix<double>, Vector<double>>
                .CreateOptimizer(OptimizerType.Adam, options));

        Assert.Contains(OptimizerType.Adam.ToString(), ex.Message);
        Assert.Contains(nameof(RpropOptimizerOptions<double, Matrix<double>, Vector<double>>), ex.Message);
    }

    [Fact]
    public void GetOptimizerType_MapsDerivedOptimizerToItsRegisteredBaseType()
    {
        var optimizer = new DerivedAdamOptimizer();

        var type = OptimizerFactory<double, Matrix<double>, Vector<double>>.GetOptimizerType(optimizer);

        Assert.Equal(OptimizerType.Adam, type);
    }

    /// <summary>
    /// An enum value with no implementation must fail with a message that says so and points somewhere useful,
    /// rather than the bare "Unknown optimizer type" that reads like the value is invalid.
    /// </summary>
    [Theory]
    [MemberData(nameof(UnimplementedOptimizerTypes))]
    public void CreateOptimizer_ForAnUnimplementedType_ExplainsWhatIsAvailable(OptimizerType type)
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            OptimizerFactory<double, Matrix<double>, Vector<double>>.CreateOptimizer(type));

        Assert.Contains(type.ToString(), ex.Message);
        // The message must enumerate real alternatives, not just say "unknown".
        Assert.Contains("Adam", ex.Message);
    }

    private sealed class DerivedAdamOptimizer
        : AdamOptimizer<double, Matrix<double>, Vector<double>>
    {
        public DerivedAdamOptimizer() : base(null) { }
    }
}
