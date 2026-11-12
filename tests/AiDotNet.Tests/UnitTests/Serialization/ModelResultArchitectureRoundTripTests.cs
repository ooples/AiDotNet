using System;
using System.Collections.Generic;
using System.IO;
using AiDotNet.ActivationFunctions;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Serialization;

/// <summary>
/// A saved-and-loaded <see cref="AiModelResult{T, TInput, TOutput}"/> must reproduce the original model exactly.
/// </summary>
/// <remarks>
/// The facade restores its model into a skeleton the JSON layer constructs with the type's DEFAULT configuration
/// (a parameterless FeedForwardNeuralNetwork declares a 128-wide input) and then pours the model's own payload into
/// it. Until that payload carried the architecture, the loaded model kept the skeleton's descriptor and rejected the
/// inputs it was trained on. With compilation on, the embedded compiled program answered Predict instead of the model
/// and hid the defect, so every case runs with compilation off as well as on.
/// </remarks>
[Collection("CompilationGlobalState")]
public class ModelResultArchitectureRoundTripTests
{
    public enum Persistence { Bytes, File }

    private static Tensor<float> Random(int seed, params int[] shape)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static void AssertClose(Tensor<float> expected, Tensor<float> actual, float tol = 1e-5f)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol, $"element {i}: {actual[i]:R} vs {expected[i]:R}");
    }

    private static AiModelResult<T, TIn, TOut> Wrap<T, TIn, TOut>(IFullModel<T, TIn, TOut> model, bool jit) =>
        new(new AiModelResultOptions<T, TIn, TOut>
        {
            OptimizationResult = new OptimizationResult<T, TIn, TOut> { BestSolution = model },
            JitCompilationConfig = jit ? JitCompilationConfig.Default : JitCompilationConfig.Disabled,
        });

    private static AiModelResult<T, TIn, TOut> RoundTrip<T, TIn, TOut>(AiModelResult<T, TIn, TOut> result, Persistence how)
    {
        var restored = new AiModelResult<T, TIn, TOut>();
        if (how == Persistence.Bytes)
        {
            using (ModelPersistenceGuard.InternalOperation())
            {
                restored.Deserialize(result.Serialize());
            }
            return restored;
        }

        string path = Path.Combine(Path.GetTempPath(), "aidotnet-roundtrip-" + Guid.NewGuid().ToString("N") + ".model");
        try
        {
            using (ModelPersistenceGuard.InternalOperation())
            {
                result.SaveModel(path);
                restored.LoadFromFile(path);
            }
            return restored;
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    private static void AssertSameArchitecture(NeuralNetworkArchitecture<float> expected, IFullModel<float, Tensor<float>, Tensor<float>>? model)
    {
        var network = Assert.IsAssignableFrom<NeuralNetworkBase<float>>(model);
        var actual = network.Architecture;
        Assert.Equal(expected.GetType(), actual.GetType());
        Assert.Equal(expected.InputType, actual.InputType);
        Assert.Equal(expected.TaskType, actual.TaskType);
        Assert.Equal(expected.Complexity, actual.Complexity);
        Assert.Equal(expected.InputSize, actual.InputSize);
        Assert.Equal(expected.InputHeight, actual.InputHeight);
        Assert.Equal(expected.InputWidth, actual.InputWidth);
        Assert.Equal(expected.InputDepth, actual.InputDepth);
        Assert.Equal(expected.OutputSize, actual.OutputSize);
        Assert.Equal(expected.GetInputShape(), actual.GetInputShape());
        Assert.Equal(expected.GetOutputShape(), actual.GetOutputShape());
        Assert.Equal(expected.Layers.Count, actual.Layers.Count);
        if (expected is TransformerArchitecture<float> et)
        {
            var at = Assert.IsType<TransformerArchitecture<float>>(actual);
            Assert.Equal(et.NumEncoderLayers, at.NumEncoderLayers);
            Assert.Equal(et.NumDecoderLayers, at.NumDecoderLayers);
            Assert.Equal(et.NumHeads, at.NumHeads);
            Assert.Equal(et.ModelDimension, at.ModelDimension);
            Assert.Equal(et.FeedForwardDimension, at.FeedForwardDimension);
            Assert.Equal(et.MaxSequenceLength, at.MaxSequenceLength);
        }
    }

    private static void AssertNeuralRoundTrip(NeuralNetworkBase<float> network, bool jit, Persistence how, params int[][] inputShapes)
    {
        var result = Wrap(network, jit);
        var inputs = new List<Tensor<float>>();
        var expected = new List<Tensor<float>>();
        for (int i = 0; i < inputShapes.Length; i++)
        {
            var x = Random(100 + i, inputShapes[i]);
            inputs.Add(x);
            expected.Add(result.Predict(x));
        }

        var restored = RoundTrip(result, how);

        AssertSameArchitecture(network.Architecture, restored.Model);
        Assert.Equal(network.LayerCount, ((NeuralNetworkBase<float>)restored.Model!).LayerCount);
        Assert.Equal(network.GetParameters().ToArray(), restored.Model!.GetParameters().ToArray());
        for (int i = 0; i < inputs.Count; i++)
        {
            AssertClose(expected[i], restored.Predict(inputs[i]));
            AssertClose(network.Predict(inputs[i]), restored.Predict(inputs[i]));
        }
    }

    // ── Families ─────────────────────────────────────────────────────────────────────────────────────────

    private static FeedForwardNeuralNetwork<float> FeedForward() => new(new NeuralNetworkArchitecture<float>(
        inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
        inputSize: 4, outputSize: 1, layers: new List<ILayer<float>>
        {
            new InputLayer<float>(4),
            new DenseLayer<float>(8, activationFunction: new ReLUActivation<float>()),
            new DenseLayer<float>(1, activationFunction: new IdentityActivation<float>()),
        }), lossFunction: new MeanSquaredErrorLoss<float>());

    private static FeedForwardNeuralNetwork<float> FeedForwardDefaultLayers() => new(new NeuralNetworkArchitecture<float>(
        inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
        complexity: NetworkComplexity.Simple, inputSize: 6, outputSize: 3));

    private static ConvolutionalNeuralNetwork<float> Conv() => new(new NeuralNetworkArchitecture<float>(
        inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
        inputHeight: 8, inputWidth: 8, inputDepth: 1, outputSize: 3, layers: new List<ILayer<float>>
        {
            new ConvolutionalLayer<float>(outputDepth: 4, kernelSize: 3, stride: 1, padding: 1,
                activationFunction: new ReLUActivation<float>()),
            new MaxPoolingLayer<float>(poolSize: 2, stride: 2),
            new FlattenLayer<float>(),
            new DenseLayer<float>(3, activationFunction: (IActivationFunction<float>?)null),
        }));

    private static LSTMNeuralNetwork<float> Lstm() => new(new NeuralNetworkArchitecture<float>(
        inputType: InputType.TwoDimensional, taskType: NeuralNetworkTaskType.Regression,
        complexity: NetworkComplexity.Simple, inputHeight: 5, inputWidth: 3, outputSize: 2),
        lossFunction: null, outputActivation: null);

    private static Transformer<float> TransformerModel()
    {
        var net = new Transformer<float>(new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional, taskType: NeuralNetworkTaskType.Regression,
            numEncoderLayers: 1, numDecoderLayers: 0, numHeads: 2, modelDimension: 8,
            feedForwardDimension: 16, inputSize: 8, outputSize: 8, maxSequenceLength: 4));
        net.Predict(Random(1, 1, 4, 8)); // lazy attention layers materialise on the first forward
        return net;
    }

    public static IEnumerable<object[]> Modes()
    {
        foreach (var jit in new[] { false, true })
            foreach (var how in new[] { Persistence.Bytes, Persistence.File })
                yield return new object[] { jit, how };
    }

    [Theory]
    [MemberData(nameof(Modes))]
    public void FeedForward_WithCustomLayers_RoundTripsArchitectureAndPredictions(bool jit, Persistence how)
        => AssertNeuralRoundTrip(FeedForward(), jit, how, new[] { 3, 4 }, new[] { 1, 4 }, new[] { 7, 4 });

    [Theory]
    [MemberData(nameof(Modes))]
    public void FeedForward_WithDefaultLayers_RoundTripsArchitectureAndPredictions(bool jit, Persistence how)
        => AssertNeuralRoundTrip(FeedForwardDefaultLayers(), jit, how, new[] { 2, 6 }, new[] { 1, 6 }, new[] { 5, 6 });

    [Theory]
    [MemberData(nameof(Modes))]
    public void ConvolutionalNetwork_RoundTripsArchitectureAndPredictions(bool jit, Persistence how)
        => AssertNeuralRoundTrip(Conv(), jit, how, new[] { 2, 1, 8, 8 }, new[] { 1, 1, 8, 8 }, new[] { 4, 1, 8, 8 });

    [Theory]
    [MemberData(nameof(Modes))]
    public void LstmNetwork_RoundTripsArchitectureAndPredictions(bool jit, Persistence how)
        => AssertNeuralRoundTrip(Lstm(), jit, how, new[] { 1, 5, 3 }, new[] { 2, 5, 3 }, new[] { 3, 5, 3 });

    [Theory]
    [MemberData(nameof(Modes))]
    public void Transformer_RoundTripsArchitectureAndPredictions(bool jit, Persistence how)
        => AssertNeuralRoundTrip(TransformerModel(), jit, how, new[] { 1, 4, 8 }, new[] { 2, 4, 8 }, new[] { 3, 4, 8 });

    [Theory]
    [MemberData(nameof(Modes))]
    public void PolynomialRegression_RoundTripsItsConfigurationAndPredictions(bool jit, Persistence how)
    {
        // Degree 3 differs from the options default, so a skeleton built with default options would predict
        // through the wrong feature expansion unless the model's own payload restores it.
        var rng = new Random(7);
        var x = new Matrix<double>(30, 2);
        var y = new Vector<double>(30);
        for (int i = 0; i < 30; i++)
        {
            double a = rng.NextDouble() * 2 - 1, b = rng.NextDouble() * 2 - 1;
            x[i, 0] = a; x[i, 1] = b;
            y[i] = 1.5 * a * a * a - 2 * b * b + a * b + 0.5;
        }
        var model = new PolynomialRegression<double>(new PolynomialRegressionOptions<double> { Degree = 3 });
        model.Train(x, y);

        var result = Wrap(model, jit);
        var probes = new[] { 1, 4, 9 };
        var expected = new List<Vector<double>>();
        var batches = new List<Matrix<double>>();
        foreach (int rows in probes)
        {
            var m = new Matrix<double>(rows, 2);
            for (int i = 0; i < rows; i++) { m[i, 0] = rng.NextDouble() * 2 - 1; m[i, 1] = rng.NextDouble() * 2 - 1; }
            batches.Add(m);
            expected.Add(result.Predict(m));
        }

        var restored = RoundTrip(result, how);

        var restoredModel = Assert.IsType<PolynomialRegression<double>>(restored.Model);
        Assert.Equal(model.GetParameters().ToArray(), restoredModel.GetParameters().ToArray());
        for (int i = 0; i < batches.Count; i++)
        {
            var actual = restored.Predict(batches[i]);
            Assert.Equal(expected[i].Length, actual.Length);
            for (int j = 0; j < actual.Length; j++) Assert.Equal(expected[i][j], actual[j], 10);
        }

        // The degree is what the feature expansion reads; it must be the trained model's, not the skeleton's.
        var degreeField = typeof(PolynomialRegression<double>).GetField("_polyOptions",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
        Assert.NotNull(degreeField);
        Assert.Equal(3, ((PolynomialRegressionOptions<double>)degreeField!.GetValue(restoredModel)!).Degree);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void StaticLoadModel_WithADefaultConstructedFactoryModel_TakesTheSavedArchitecture(bool jit)
    {
        var network = FeedForward();
        var result = Wrap(network, jit);
        var x = Random(21, 6, 4);
        var before = result.Predict(x);

        string path = Path.Combine(Path.GetTempPath(), "aidotnet-roundtrip-" + Guid.NewGuid().ToString("N") + ".model");
        try
        {
            AiModelResult<float, Tensor<float>, Tensor<float>> loaded;
            using (ModelPersistenceGuard.InternalOperation())
            {
                result.SaveModel(path);
                loaded = AiModelResult<float, Tensor<float>, Tensor<float>>.LoadModel(
                    path, _ => new FeedForwardNeuralNetwork<float>());
            }

            AssertSameArchitecture(network.Architecture, loaded.Model);
            AssertClose(before, loaded.Predict(x));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Network_DeserializedIntoADefaultSkeleton_TakesTheSavedArchitecture()
    {
        // The facade's path in miniature: a model's own payload restored into a default-constructed skeleton.
        var original = FeedForward();
        var x = Random(9, 3, 4);
        var before = original.Predict(x);

        var target = new FeedForwardNeuralNetwork<float>();
        Assert.Equal(128, target.Architecture.InputSize);
        using (ModelPersistenceGuard.InternalOperation())
        using (SerializedArchitectureAdoption.Begin())
        {
            target.Deserialize(original.Serialize());
        }

        AssertSameArchitecture(original.Architecture, target);
        AssertClose(before, target.Predict(x));
    }

    [Fact]
    public void Network_DeserializedIntoAnInstanceTheCallerBuilt_KeepsItsOwnArchitecture()
    {
        // Outside the facade's skeleton restore, the instance's architecture is the caller's: a subclass may depend on
        // it (a code model's tokenizer) or validate the payload against it, so the payload must not overwrite it.
        var original = FeedForward();
        var target = new FeedForwardNeuralNetwork<float>();
        using (ModelPersistenceGuard.InternalOperation())
        {
            target.Deserialize(original.Serialize());
        }

        Assert.Equal(128, target.Architecture.InputSize);
        Assert.Equal(original.Layers.Count, target.Layers.Count);
    }

    [Fact]
    public void Network_FromANewerFormatVersion_IsRejectedInsteadOfMisread()
    {
        // A later format inserts fields this reader does not know, so every field after them would be misread: the
        // header's version must be checked against the newest this library writes.
        var bytes = FeedForward().Serialize();
        byte[] magic = BitConverter.GetBytes(0x4E444941); // "AIDN", the network payload header
        int at = -1;
        for (int i = 0; i + 8 <= bytes.Length && at < 0; i++)
            if (bytes[i] == magic[0] && bytes[i + 1] == magic[1] && bytes[i + 2] == magic[2] && bytes[i + 3] == magic[3])
                at = i;
        Assert.True(at >= 0, "the serialized network carries no payload header");
        int written = BitConverter.ToInt32(bytes, at + 4);
        Array.Copy(BitConverter.GetBytes(written + 1), 0, bytes, at + 4, 4);

        var target = new FeedForwardNeuralNetwork<float>();
        using (ModelPersistenceGuard.InternalOperation())
        {
            var error = Assert.Throws<InvalidDataException>(() => target.Deserialize(bytes));
            Assert.Contains($"version {written + 1}", error.Message);
        }
    }
}
