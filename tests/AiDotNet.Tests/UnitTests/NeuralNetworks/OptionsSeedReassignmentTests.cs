using System;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Assigning <c>Options</c> applies its seed by restarting the layer initialization scope. A model that assigns
/// the SAME seeded options again after building some layers must not restart it: the layers built afterwards
/// would draw the seeds the earlier ones already used and initialise identically.
/// </summary>
public class OptionsSeedReassignmentTests
{
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Input, BatchOptional = true)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Output, BatchOptional = true)]
    private sealed class ReassignsOptions : NeuralNetworkBase<double>
    {
        private readonly NeuralNetworkOptions _options;

        public ReassignsOptions(NeuralNetworkArchitecture<double> architecture, int seed)
            : base(architecture, new MeanSquaredErrorLoss<double>())
        {
            _options = new NeuralNetworkOptions { Seed = seed };
            Options = _options;
            InitializeLayers();
        }

        protected override void InitializeLayers()
        {
            Layers.Add(new FullyConnectedLayer<double>(4, 4, new AiDotNet.ActivationFunctions.IdentityActivation<double>()));

            // The same seeded options, assigned again part-way through construction.
            Options = _options;
            Layers.Add(new FullyConnectedLayer<double>(4, 4, new AiDotNet.ActivationFunctions.IdentityActivation<double>()));
        }

        public override IFullModel<double, Tensor<double>, Tensor<double>> DeepCopy()
            => new ReassignsOptions(Architecture, _options.Seed ?? 0);

        public override ModelMetadata<double> GetModelMetadata() => new()
        {
            Name = nameof(ReassignsOptions),
            Description = "Test double that reassigns its seeded options between two layers.",
        };
    }

    [Fact]
    public void ReassigningTheSameSeededOptions_DoesNotRepeatTheLayerSeeds()
    {
        var model = new ReassignsOptions(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
                inputSize: 4, outputSize: 4),
            seed: 2290);
        model.MaterializeParameters();

        var first = model.Layers[0].GetParameters().ToArray();
        var second = model.Layers[1].GetParameters().ToArray();

        Assert.Equal(first.Length, second.Length);
        Assert.NotEqual(first, second);
    }
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Input, BatchOptional = true)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Output, BatchOptional = true)]
    private sealed class SuppliedWithOptions : NeuralNetworkBase<double>
    {
        public SuppliedWithOptions(NeuralNetworkArchitecture<double> architecture, params int?[] optionSeeds)
            : base(architecture, new MeanSquaredErrorLoss<double>())
        {
            foreach (var seed in optionSeeds) Options = new NeuralNetworkOptions { Seed = seed };
            InitializeLayers();
        }

        protected override void InitializeLayers() => Layers.AddRange(Architecture.Layers ?? new System.Collections.Generic.List<ILayer<double>>());

        public override IFullModel<double, Tensor<double>, Tensor<double>> DeepCopy() => new SuppliedWithOptions(Architecture);

        public override ModelMetadata<double> GetModelMetadata() => new()
        {
            Name = nameof(SuppliedWithOptions),
            Description = "Test double that assigns a sequence of options before running its caller-built layers.",
        };
    }

    private static (SuppliedWithOptions Model, FullyConnectedLayer<double> Layer) Supplied(int? layerSeed, params int?[] optionSeeds)
    {
        var layer = new FullyConnectedLayer<double>(4, 4, new AiDotNet.ActivationFunctions.IdentityActivation<double>());
        if (layerSeed is int seed) layer.RandomSeed = seed;
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4, outputSize: 4, layers: new System.Collections.Generic.List<ILayer<double>> { layer });
        return (new SuppliedWithOptions(architecture, optionSeeds), layer);
    }

    /// <summary>
    /// Options without a seed that replace seeded ones leave the model unseeded: its caller-built layer no longer
    /// carries the seed the replaced options gave it.
    /// </summary>
    [Fact]
    public void UnseededOptionsReplacingSeededOnes_DropTheEarlierSeed()
    {
        var (seeded, seededLayer) = Supplied(null, 5);
        var (replaced, replacedLayer) = Supplied(null, 5, null);

        Assert.NotNull(seededLayer.RandomSeed);
        Assert.NotEqual(seededLayer.RandomSeed, replacedLayer.RandomSeed);
        GC.KeepAlive(seeded);
        GC.KeepAlive(replaced);
    }

    /// <summary>A seed the caller set on a layer they built survives the seed wiring of the first training step.</summary>
    [Fact]
    public void ACallerChosenLayerSeed_SurvivesTraining()
    {
        var (model, layer) = Supplied(777, 5);
        var input = new Tensor<double>(new[] { 1, 4 });
        var target = new Tensor<double>(new[] { 1, 4 });
        for (int i = 0; i < 4; i++) { input[i] = 0.1 * (i + 1); target[i] = 0.2 * i; }

        model.Train(input, target);

        Assert.Equal(777, layer.RandomSeed);
    }

    /// <summary>
    /// The pattern most models follow: options kept in a field of the model's own, exposed through GetOptions(),
    /// and never assigned to the base's Options. The base reads the seed through the layer-initialization deferral.
    /// </summary>
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Input, BatchOptional = true)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Output, BatchOptional = true)]
    private sealed class KeepsOwnOptions : NeuralNetworkBase<double>
    {
        private readonly NeuralNetworkOptions _options;

        public KeepsOwnOptions(NeuralNetworkArchitecture<double> architecture, int? seed)
            : base(architecture, new MeanSquaredErrorLoss<double>())
        {
            _options = new NeuralNetworkOptions { Seed = seed };
            InitializeLayers();
        }

        public override ModelOptions GetOptions() => _options;

        protected override void InitializeLayers()
        {
            Layers.Add(new FullyConnectedLayer<double>(4, 8, new AiDotNet.ActivationFunctions.IdentityActivation<double>()));
            Layers.Add(new DropoutLayer<double>(0.5));
            Layers.Add(new FullyConnectedLayer<double>(8, 4, new AiDotNet.ActivationFunctions.IdentityActivation<double>()));
        }

        public override IFullModel<double, Tensor<double>, Tensor<double>> DeepCopy()
            => new KeepsOwnOptions(Architecture, _options.Seed);

        public override ModelMetadata<double> GetModelMetadata() => new()
        {
            Name = nameof(KeepsOwnOptions),
            Description = "Test double that keeps its seeded options to itself.",
        };
    }

    private static KeepsOwnOptions BuildKeepsOwnOptions(int? seed)
    {
        var model = new KeepsOwnOptions(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
                inputSize: 4, outputSize: 4),
            seed);
        model.MaterializeParameters();
        return model;
    }

    // Draws from the thread's shared generator, which unseeded initialization falls back to, so a model that ignored
    // its seed would initialise differently after this.
    private static void AdvanceTheSharedGenerator()
    {
        for (int i = 0; i < 1000; i++)
            AiDotNet.Tensors.Helpers.RandomHelper.ThreadSafeRandom.NextDouble();
    }

    [Fact]
    public void OptionsKeptByTheModel_SeedItsInitialWeights()
    {
        var first = BuildKeepsOwnOptions(2290).GetParameters().ToArray();
        AdvanceTheSharedGenerator();
        var second = BuildKeepsOwnOptions(2290).GetParameters().ToArray();
        var other = BuildKeepsOwnOptions(2291).GetParameters().ToArray();

        Assert.Equal(first, second);
        Assert.NotEqual(first, other);
    }

    [Fact]
    public void OptionsKeptByTheModel_SeedItsDropoutStream()
    {
        double[] Trained()
        {
            var model = BuildKeepsOwnOptions(2290);
            var input = new Tensor<double>(new[] { 2, 4 });
            var target = new Tensor<double>(new[] { 2, 4 });
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = Math.Sin(0.7 * i);
                target[i] = Math.Cos(0.3 * i);
            }
            for (int step = 0; step < 3; step++)
                model.Train(input, target);
            return model.GetParameters().ToArray();
        }

        var first = Trained();
        AdvanceTheSharedGenerator();
        Assert.Equal(first, Trained());
    }

    /// <summary>
    /// Keeps its only layer outside Layers and builds it before its seeded options exist, so no seed request comes
    /// after the options do and the deferral is still pending when construction ends.
    /// </summary>
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Input, BatchOptional = true)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features,
        Direction = TensorLayoutDirection.Output, BatchOptional = true)]
    private sealed class KeepsLayersOffList : NeuralNetworkBase<double>
    {
        private readonly FullyConnectedLayer<double> _head;
        private readonly NeuralNetworkOptions _options;

        public KeepsLayersOffList(NeuralNetworkArchitecture<double> architecture, int? seed)
            : base(architecture, new MeanSquaredErrorLoss<double>())
        {
            _head = new FullyConnectedLayer<double>(4, 4, new AiDotNet.ActivationFunctions.IdentityActivation<double>());
            _options = new NeuralNetworkOptions { Seed = seed };
        }

        public override ModelOptions GetOptions() => _options;

        protected override IEnumerable<LayerBase<double>?> GetExtraTrainableLayers()
        {
            yield return _head;
        }

        protected override void InitializeLayers()
        {
        }

        public override IFullModel<double, Tensor<double>, Tensor<double>> DeepCopy()
            => new KeepsLayersOffList(Architecture, _options.Seed);

        public override ModelMetadata<double> GetModelMetadata() => new()
        {
            Name = nameof(KeepsLayersOffList),
            Description = "Test double that holds its layer outside Layers.",
        };
    }

    [Fact]
    public void AFinishedModelWithOffListLayers_DoesNotSeedALayerBuiltAfterIt()
    {
        _ = new KeepsLayersOffList(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
                inputSize: 4, outputSize: 4),
            2290);
        var unrelated = new FullyConnectedLayer<double>(4, 4, new AiDotNet.ActivationFunctions.IdentityActivation<double>());

        // Handing the finished model's seed to the scope would give this layer the first draw of that seed's stream.
        Assert.NotEqual(AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(2290).Next(), unrelated.RandomSeed);
    }

    /// <summary>
    /// The base owns a network's options: a model hands them over with <c>Options = _options</c> and inherits
    /// <c>GetOptions()</c>. About a thousand models instead repeated <c>GetOptions() => _options</c> and many of them
    /// never assigned <c>Options</c>, so the base never saw their seed. An override that only returns the model's
    /// options is that defect waiting to recur; one with real logic belongs in a base, not a leaf model.
    /// </summary>
    [Fact]
    public void NetworkModels_InheritGetOptionsFromTheBase()
    {
        var redeclaring = typeof(NeuralNetworkBase<double>).Assembly.GetTypes()
            .Where(type => type != typeof(NeuralNetworkBase<>) && DerivesFromNeuralNetworkBase(type))
            .Where(type => type.GetMethod(
                nameof(NeuralNetworkBase<double>.GetOptions),
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public
                    | System.Reflection.BindingFlags.DeclaredOnly,
                null, Type.EmptyTypes, null) is not null)
            .Select(type => type.FullName ?? type.Name)
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToArray();

        Assert.True(redeclaring.Length == 0,
            "These network types declare their own GetOptions(); assign Options in the constructor instead: "
            + string.Join(", ", redeclaring));
    }

    private static bool DerivesFromNeuralNetworkBase(Type type)
    {
        for (var current = type.BaseType; current is not null; current = current.BaseType)
            if (current.IsGenericType && current.GetGenericTypeDefinition() == typeof(NeuralNetworkBase<>))
                return true;
        return false;
    }
}