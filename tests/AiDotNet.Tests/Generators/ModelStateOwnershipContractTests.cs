using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;

namespace AiDotNet.Tests.Generators;

public sealed class ModelStateOwnershipContractTests
{
    public enum StorageMutability { Mutable, Readonly }
    public enum LegacyEnvelope { Absent, MissingRawFields }

    public interface IStateProbe : IDisposable
    {
        NeuralNetworkBase<float> GetModel();
        void SetRawState();
        float[] ReadRawState();
        double ReadScale();
        object[] ReadRawReferences();
    }

    public ModelStateOwnershipContractTests() => TestModuleInitializer.EnsureInitialized();

    [Theory]
    [InlineData(StorageMutability.Mutable)]
    [InlineData(StorageMutability.Readonly)]
    public void NeuralStateGenerationPersistsRawStorageButNotCanonicalLayerAliases(StorageMutability mutability)
    {
        var (_, generated) = CompileProbe(mutability);
        Assert.Contains("RawStateNetwork._tensor", generated, StringComparison.Ordinal);
        Assert.Contains("RawStateNetwork._vector", generated, StringComparison.Ordinal);
        Assert.Contains("RawStateNetwork._matrix", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("RawStateNetwork._canonical", generated, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(StorageMutability.Mutable)]
    [InlineData(StorageMutability.Readonly)]
    public void ActualGeneratedNeuralStateRoundTripsTensorVectorMatrixAndNativePrecision(StorageMutability mutability)
    {
        var (factory, _) = CompileProbe(mutability);
        using var source = factory();
        source.SetRawState();
        var expected = source.ReadRawState();
        double expectedScale = source.ReadScale();
        var bytes = source.GetModel().Serialize();
        using var restored = factory();
        var references = restored.ReadRawReferences();
        _ = restored.GetModel().GetParameters(); // Exercise a previously materialized parameter layout too.
        Assert.False(expected.SequenceEqual(restored.ReadRawState()));
        restored.GetModel().Deserialize(bytes);
        Assert.Equal(expected, restored.ReadRawState());
        Assert.Equal(expectedScale, restored.ReadScale());
        Assert.Equal(source.GetModel().GetParameters().ToArray(), restored.GetModel().GetParameters().ToArray());
        Assert.Single(restored.GetModel().Layers);
        if (mutability == StorageMutability.Readonly)
            for (int index = 0; index < references.Length; index++)
                Assert.Same(references[index], restored.ReadRawReferences()[index]);
    }

    [Theory]
    [InlineData(StorageMutability.Mutable, LegacyEnvelope.Absent)]
    [InlineData(StorageMutability.Readonly, LegacyEnvelope.Absent)]
    [InlineData(StorageMutability.Mutable, LegacyEnvelope.MissingRawFields)]
    [InlineData(StorageMutability.Readonly, LegacyEnvelope.MissingRawFields)]
    public void LegacyCheckpointsKeepConstructorValuesForRawFieldsTheyNeverStored(
        StorageMutability mutability, LegacyEnvelope envelope)
    {
        var (factory, _) = CompileProbe(mutability);
        using var source = factory();
        source.SetRawState();
        // Strip the additive envelope without altering the real layer-format payload. This
        // deliberately models missing historical state, not recovery of values never saved.
        byte[] legacy = ModelStateEnvelope.Extract(new ModelStateRegistry<float>(), source.GetModel().Serialize());
        if (envelope == LegacyEnvelope.MissingRawFields)
        {
            var partialState = new ModelStateRegistry<float>();
            partialState.DeclareDouble("RawStateNetwork._scale", source.ReadScale, _ => { });
            legacy = ModelStateEnvelope.Append(partialState, legacy);
        }

        using var restored = factory();
        var constructorValues = restored.ReadRawState();
        var constructorReferences = restored.ReadRawReferences();
        restored.GetModel().Deserialize(legacy);
        Assert.Equal(constructorValues, restored.ReadRawState());
        Assert.Equal(envelope == LegacyEnvelope.MissingRawFields ? source.ReadScale() : 0.25, restored.ReadScale());
        Assert.Equal(source.GetModel().Layers[0].GetParameters().ToArray(),
            restored.GetModel().Layers[0].GetParameters().ToArray());
        for (int index = 0; index < constructorReferences.Length; index++)
            Assert.Same(constructorReferences[index], restored.ReadRawReferences()[index]);
    }

    [Fact]
    public void FlatCheckpointOwnerPersistsRawStorageExactlyOnceThroughItsParameterRegistry()
    {
        const string source = """
            using AiDotNet.Attributes;
            using AiDotNet.Models;
            using AiDotNet.Tensors.LinearAlgebra;
            namespace AiDotNet.Tests.GeneratedOwnership;
            public partial class FlatStateModel : VectorModel<float>
            {
                [TrainableParameter] private readonly Tensor<float> _tensor = new(new[] { 3 });
                [TrainableParameter] private readonly Vector<float> _vector = new(2);
                [TrainableParameter] private readonly Matrix<float> _matrix = new(2, 2);
                public FlatStateModel() : base(new Vector<float>(3)) { }
            }
            """;
        var (assembly, generated) = CompileSource(source);
        Assert.DoesNotContain("FlatStateModel._tensor", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("FlatStateModel._vector", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("FlatStateModel._matrix", generated, StringComparison.Ordinal);
        Type type = assembly.GetType("AiDotNet.Tests.GeneratedOwnership.FlatStateModel")
            ?? throw new InvalidOperationException("The generated flat-state model was not compiled.");
        using var model = Assert.IsAssignableFrom<VectorModel<float>>(Activator.CreateInstance(type));
        var values = model.GetParameters();
        Assert.True(values.Length >= 12, "The flat registry must actually contain all three raw storage fields.");
        for (int index = 0; index < values.Length; index++) values[index] = 20 + index;
        model.SetParameters(values);
        using var restored = Assert.IsAssignableFrom<VectorModel<float>>(Activator.CreateInstance(type));
        Assert.False(model.GetParameters().ToArray().SequenceEqual(restored.GetParameters().ToArray()));
        restored.Deserialize(model.Serialize());
        Assert.Equal(model.GetParameters().ToArray(), restored.GetParameters().ToArray());
    }

    [Fact]
    public void NonNeuralNonpartialReadonlyBuffersRetainTheirExistingPersistenceBoundary()
    {
        const string source = """
            using AiDotNet.Attributes;
            using AiDotNet.Models;
            using AiDotNet.Tensors.LinearAlgebra;
            namespace AiDotNet.Tests.GeneratedOwnership;
            // Like WeightedRegression's per-observation training buffer, this is not the
            // neural layer-checkpoint trunk corrected by this change. It is not migrated here.
            public class NonNeuralBufferModel : VectorModel<float>
            {
                [Buffer] private readonly Vector<float> _sampleWeights = new(3);
                public NonNeuralBufferModel() : base(new Vector<float>(3)) { }
            }
            """;
        var (_, generated) = CompileSource(source);
        Assert.DoesNotContain("NonNeuralBufferModel._sampleWeights", generated, StringComparison.Ordinal);
    }

    private static (Func<IStateProbe> Factory, string StateSource) CompileProbe(StorageMutability mutability)
    {
        string fieldModifier = mutability == StorageMutability.Readonly ? "readonly " : string.Empty;
        string source = $$"""
            using System;
            using System.Linq;
            using AiDotNet.Attributes;
            using AiDotNet.Enums;
            using AiDotNet.LossFunctions;
            using AiDotNet.NeuralNetworks;
            using AiDotNet.NeuralNetworks.Layers;
            using AiDotNet.Tensors.LinearAlgebra;

            namespace AiDotNet.Tests.GeneratedOwnership;

            public partial class RawStateNetwork : NeuralNetworkBase<float>,
                AiDotNet.Tests.Generators.ModelStateOwnershipContractTests.IStateProbe
            {
                [TrainableParameter] private {{fieldModifier}}Tensor<float> _tensor = new(new[] { 3 });
                [TrainableParameter] private {{fieldModifier}}Vector<float> _vector = new(2);
                [TrainableParameter] private {{fieldModifier}}Matrix<float> _matrix = new(2, 2);
                [TrainableParameter] private double _scale = 0.25;
                private readonly DenseLayer<float> _canonical;

                public RawStateNetwork() : base(new NeuralNetworkArchitecture<float>(
                    inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
                    inputSize: 4, outputSize: 1), new MeanSquaredErrorLoss<float>())
                {
                    _canonical = new DenseLayer<float>(1, (AiDotNet.Interfaces.IActivationFunction<float>?)null);
                    Layers.Add(_canonical);
                    _ = _canonical.Forward(new Tensor<float>(new[] { 1, 4 }));
                }

                public NeuralNetworkBase<float> GetModel() => this;
                protected override void InitializeLayers() { }
                public override AiDotNet.Models.ModelMetadata<float> GetModelMetadata() => new() { Name = nameof(RawStateNetwork) };
                public void SetRawState()
                {
                    for (int index = 0; index < _tensor.Length; index++) _tensor[index] = 10.0f + index;
                    for (int index = 0; index < _vector.Length; index++) _vector[index] = 20.0f + index;
                    for (int row = 0; row < 2; row++)
                        for (int column = 0; column < 2; column++) _matrix[row, column] = 30.0f + row * 2 + column;
                    _scale = 16777217.125;
                }
                public float[] ReadRawState() => _tensor.ToArray().Concat(_vector.ToArray())
                    .Concat(new[] { _matrix[0, 0], _matrix[0, 1], _matrix[1, 0], _matrix[1, 1] }).ToArray();
                public double ReadScale() => _scale;
                public object[] ReadRawReferences() => new object[] { _tensor, _vector, _matrix };
            }
            """;

        var (loaded, state) = CompileSource(source);
        var type = loaded.GetType("AiDotNet.Tests.GeneratedOwnership.RawStateNetwork", throwOnError: true)
            ?? throw new InvalidOperationException("The real generated state probe was not emitted.");
        IStateProbe Factory() => Activator.CreateInstance(type) as IStateProbe
            ?? throw new InvalidOperationException("The generated state probe has the wrong runtime contract.");
        return (Factory, state);
    }

    private static (Assembly Assembly, string StateSource) CompileSource(string source)
    {
        var paths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var references = new List<MetadataReference>();
        var trusted = AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") as string;
        if (trusted is not null)
            foreach (string path in trusted.Split(Path.PathSeparator))
                if (paths.Add(path)) references.Add(MetadataReference.CreateFromFile(path));
        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
            if (!assembly.IsDynamic && !string.IsNullOrEmpty(assembly.Location) && paths.Add(assembly.Location))
                references.Add(MetadataReference.CreateFromFile(assembly.Location));
        // Generated in-library models use internal ownership helpers. Reuse the repository's
        // existing test-friend identity without changing or widening that production boundary.
        var compilation = CSharpCompilation.Create("AiDotNetTestConsole",
            new[] { CSharpSyntaxTree.ParseText(source) }, references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, nullableContextOptions: NullableContextOptions.Enable));
        var stateGenerator = new AiDotNet.Generators.ModelStateGenerator().AsSourceGenerator();
        GeneratorDriver driver = CSharpGeneratorDriver.Create(
            new AiDotNet.Generators.ModelParameterGenerator().AsSourceGenerator(), stateGenerator);
        driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out var output, out var generatorDiagnostics);
        Assert.Empty(generatorDiagnostics.Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        Assert.Empty(output.GetDiagnostics().Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));
        using var stream = new MemoryStream();
        var emitted = output.Emit(stream);
        Assert.True(emitted.Success, string.Join(Environment.NewLine, emitted.Diagnostics));
        var loaded = Assembly.Load(stream.ToArray());
        string state = string.Join(Environment.NewLine, Assert.Single(driver.GetRunResult().Results,
                result => ReferenceEquals(result.Generator, stateGenerator)).GeneratedSources
            .Select(result => result.SourceText.ToString()));
        return (loaded, state);
    }
}
