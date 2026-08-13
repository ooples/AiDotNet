using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public class InputContractTests
{
    [Fact]
    public async Task UnresolvedIndexRange_NeverBecomesContinuous()
    {
        await Task.Yield();
        var domain = LayerInputDomain.Indices(0);

        Assert.Equal(LayerInputDomainKind.Deferred, domain.Kind);
        Assert.False(domain.IsResolved);
        Assert.Throws<InputContractBindingException>(() =>
            InputContractTensorFactory.CreateValid<double>([4], domain, new Random(1)));
    }

    [Fact]
    public async Task PreserveRelationship_IsNotAnAcceptEverythingWildcard()
    {
        await Task.Yield();
        var compatibility = LayerInputDomain.Indices(32)
            .CompatibilityWith(LayerInputDomain.Preserve("input"));

        Assert.Equal(LayerInputDomainCompatibility.Deferred, compatibility);
        Assert.False(LayerInputDomain.Indices(32).Accepts(LayerInputDomain.Preserve("input")));
    }

    [Fact]
    public async Task IntegerCompatibility_ProvesRangeContainment()
    {
        await Task.Yield();

        Assert.Equal(
            LayerInputDomainCompatibility.Compatible,
            LayerInputDomain.Indices(64).CompatibilityWith(LayerInputDomain.Indices(32)));
        Assert.Equal(
            LayerInputDomainCompatibility.Incompatible,
            LayerInputDomain.Indices(32).CompatibilityWith(LayerInputDomain.Indices(64)));
    }

    [Fact]
    public async Task ValidTensorFactory_AndRuntimeValidator_AreInverses()
    {
        await Task.Yield();
        var domain = LayerInputDomain.Indices(17);
        var tensor = InputContractTensorFactory.CreateValid<double>([2, 8], domain, new Random(7));

        InputContractValidator.ValidateValues(tensor, domain, "Lookup", "token_ids");
        for (int i = 0; i < tensor.Length; i++) Assert.InRange(tensor[i], 0, 16);
    }

    [Fact]
    public async Task NegativeSynthesis_IsRejectedBySameValidator()
    {
        await Task.Yield();
        var domain = LayerInputDomain.BooleanMask;
        var invalid = InputContractTensorFactory.CreateInvalid<double>([4], domain);

        var exception = Assert.Throws<InputContractViolationException>(() =>
            InputContractValidator.ValidateValues(invalid, domain, "Attention", "mask"));
        Assert.Contains("only 0 or 1", exception.Message);
    }

    [Fact]
    public async Task BoundContract_RejectsGeometryBeforeForward()
    {
        await Task.Yield();
        var port = new LayerPort(
            "input",
            [2, 6],
            ShapeConstraint: new PortShapeConstraint
            {
                ExactRank = 2,
                MinimumAxisSizes = new[] { 1, 8 },
                AxisDivisors = new[] { 1, 4 }
            });
        var manifest = new InputContractManifest("GeometryModel", [port]);

        var contract = manifest.Bind([2, 6]);

        Assert.Equal(InputContractReadiness.Invalid, contract.Readiness);
        Assert.Contains(contract.Reasons, reason => reason.Contains("axis 1"));
        Assert.Throws<InputContractBindingException>(() => contract.RequireReady());
    }

    [Fact]
    public async Task CallerGeometry_IsInvalid_WhileUnresolvedDeclarationsRemainDeferred()
    {
        await Task.Yield();
        var manifest = new InputContractManifest(
            "StrictBoundary",
            [new LayerPort("input", [-1], ValueDomain: LayerInputDomain.Indices(0))]);

        var invalid = manifest.Bind([0, 5]);
        Assert.Equal(InputContractReadiness.Invalid, invalid.Readiness);
        Assert.Throws<InputContractViolationException>(() =>
            invalid.Validate(new Tensor<double>([0, 5])));

        var deferred = manifest.Bind([2, 5]);
        Assert.Equal(InputContractReadiness.Deferred, deferred.Readiness);
        Assert.Throws<InputContractBindingException>(() => deferred.RequireReady());
    }

    [Fact]
    public async Task AlternativeVariants_AreResolvedFromNamedPorts()
    {
        await Task.Yield();
        var manifest = new InputContractManifest(
            "AlternativeModel",
            [
                new LayerPort("features", [4], Variant: "features"),
                new LayerPort(
                    "token_ids",
                    [4],
                    ValueDomain: LayerInputDomain.Indices(32),
                    Role: TensorPortRole.TokenIds,
                    Variant: "tokens")
            ]);

        Assert.Equal("features", manifest.ResolveVariant(["features"]));
        Assert.Equal("tokens", manifest.ResolveVariant(["token_ids"]));
    }

    [Fact]
    public async Task FloatIndexCardinality_MustBeExactlyRepresentable()
    {
        await Task.Yield();
        var tensor = new Tensor<float>([1]);

        var exception = Assert.Throws<InputContractBindingException>(() =>
            InputContractValidator.ValidateValues(
                tensor,
                LayerInputDomain.Indices(16_777_217),
                "HugeLookup",
                "token_ids"));
        Assert.Contains("represent consecutive integers exactly", exception.Message);
    }

    [Fact]
    public async Task ShapeResolver_AppliesRankAxisAndElementRulesTogether()
    {
        await Task.Yield();
        var constraint = new ModelInputShapeConstraint(
            MinimumRank: 3,
            MinimumElementCount: 80,
            MinimumAxisSizes: new[] { 1, 3, 5 },
            AxisDivisors: new[] { 1, 2, 4 });

        int[] resolved = InputContractShapeResolver.Conform([3, 2], constraint);

        Assert.Equal([1, 4, 20], resolved);
    }

    [Fact]
    public async Task ShapeResolver_DoesNotDiscardNonUnitAxesToSatisfyExactRank()
    {
        await Task.Yield();

        var exception = Assert.Throws<InputContractBindingException>(() =>
            InputContractShapeResolver.Conform(
                [2, 3, 4],
                new ModelInputShapeConstraint(0, 0, ExactRank: 2)));

        Assert.Contains("without discarding a non-unit leading axis", exception.Message);
    }

    [Fact]
    public async Task NamedInputSynthesis_CreatesAllRequiredExternalPortsOnly()
    {
        await Task.Yield();
        var manifest = new InputContractManifest(
            "Attention",
            [
                new LayerPort("query", [2, 4]),
                new LayerPort(
                    "mask",
                    [2, 4],
                    ValueDomain: LayerInputDomain.BooleanMask,
                    Role: TensorPortRole.Mask),
                new LayerPort(
                    "cache",
                    [2, 4],
                    Required: false,
                    Source: TensorPortSource.Internal)
            ]);
        var contract = manifest.Bind([2, 4]);

        var inputs = InputContractTensorFactory.CreateValidInputs<double>(
            contract,
            new Random(9));

        Assert.Equal(2, inputs.Count);
        Assert.Contains("query", inputs.Keys);
        Assert.Contains("mask", inputs.Keys);
        Assert.DoesNotContain("cache", inputs.Keys);
    }

    [Fact]
    public async Task NamedBinding_ResolvesEachPortShapeAndEnforcesRelations()
    {
        await Task.Yield();
        var manifest = new InputContractManifest(
            "Decoder",
            [
                new LayerPort("decoder_input", [-1], Variant: "named"),
                new LayerPort(
                    "encoder_output",
                    [-1],
                    Source: TensorPortSource.External,
                    Variant: "named",
                    ShapeConstraint: new PortShapeConstraint { SameShapeAs = "decoder_input" })
            ]);
        var contract = manifest.Bind(
            new Dictionary<string, int[]>
            {
                ["decoder_input"] = [2, 4],
                ["encoder_output"] = [2, 4]
            },
            "named");

        contract.RequireReady();
        Assert.All(contract.InputPorts, port => Assert.Equal([2, 4], port.Shape));

        var invalid = new Dictionary<string, Tensor<double>>
        {
            ["decoder_input"] = new([2, 4]),
            ["encoder_output"] = new([3, 4])
        };
        var error = Assert.Throws<InputContractViolationException>(() => contract.Validate(invalid));
        Assert.Contains("same shape", error.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task CustomDomain_IsFailClosedAndUsesOneProviderForAllOperations()
    {
        await Task.Yield();
        var domain = LayerInputDomain.Custom(UnitIntervalProvider.ProviderKey);
        Assert.False(domain.IsResolved);

        using (InputDomainProviderRegistry.Register(new UnitIntervalProvider()))
        {
            Assert.True(domain.IsResolved);
            var valid = InputContractTensorFactory.CreateValid<double>([4], domain, new Random(3));
            InputContractValidator.ValidateValues(valid, domain, "Probability", "weights");

            var nearby = InputContractTensorFactory.CreateNearby(valid, domain);
            InputContractValidator.ValidateValues(nearby, domain, "Probability", "weights");

            var invalid = InputContractTensorFactory.CreateInvalid<double>([4], domain);
            Assert.Throws<InputContractViolationException>(() =>
                InputContractValidator.ValidateValues(invalid, domain, "Probability", "weights"));
        }

        Assert.False(domain.IsResolved);
    }

    private sealed class UnitIntervalProvider : IInputDomainProvider
    {
        public const string ProviderKey = "tests.unit-interval";
        public string Key => ProviderKey;

        public LayerInputDomainCompatibility CompatibilityWith(LayerInputDomain producer) =>
            producer.Kind == LayerInputDomainKind.Custom && producer.Detail == ProviderKey
                ? LayerInputDomainCompatibility.Compatible
                : LayerInputDomainCompatibility.Incompatible;

        public void Validate<T>(Tensor<T> input, string ownerName, string portName)
        {
            var operations = MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < input.Length; i++)
            {
                double value = operations.ToDouble(input[i]);
                if (!double.IsNaN(value) && value >= 0.0 && value <= 1.0) continue;
                throw new InputContractViolationException(
                    $"{ownerName}.{portName} requires values in [0, 1], but element {i} is {value}.",
                    portName);
            }
        }

        public Tensor<T> CreateValid<T>(int[] shape, Random random)
        {
            var tensor = new Tensor<T>(shape);
            var operations = MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < tensor.Length; i++)
                tensor[i] = operations.FromDouble(random.NextDouble());
            return tensor;
        }

        public Tensor<T> CreateNearby<T>(Tensor<T> input, double epsilon)
        {
            var nearby = new Tensor<T>(input.Shape.ToArray());
            var operations = MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < input.Length; i++)
                nearby[i] = operations.FromDouble(
                    Math.Min(1.0, operations.ToDouble(input[i]) + epsilon));
            return nearby;
        }

        public Tensor<T> CreateInvalid<T>(int[] shape)
        {
            var tensor = new Tensor<T>(shape);
            var operations = MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < tensor.Length; i++) tensor[i] = operations.FromDouble(2.0);
            return tensor;
        }
    }
}
