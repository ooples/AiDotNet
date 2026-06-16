using AiDotNet;
using AiDotNet.ActivationFunctions;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Repro + regression test for #1629 — every layer in <c>src/NeuralNetworks/Layers/</c>
/// that accepted an optional <c>IActivationFunction&lt;T&gt;? activationFunction = null</c>
/// silently substituted a <see cref="ReLUActivation{T}"/> when callers passed
/// <c>null</c> or omitted the argument.
///
/// <para><b>User-visible symptom:</b> A regression model written as
/// <c>new DenseLayer&lt;double&gt;(outputSize: 1)</c> with a <c>// linear output for regression</c>
/// comment would silently get a ReLU output layer. Once training nudged the
/// pre-activation negative, ReLU clamped every test prediction to exact 0.0 —
/// the gradient through dead ReLU is also 0, so the model never recovered.
/// On the AiDotNet facade walkthrough Example 2 (Sales Forecasting, Regression,
/// MSE loss) this produced <c>pred=(0.000000, 0.000000, 0.000000, 0.000000, 0.000000)</c>
/// with RMSE 0.685.</para>
///
/// <para><b>Scope of fix:</b> all 13 layer types in <c>src/NeuralNetworks/Layers/</c>
/// that took an optional activation: <see cref="DenseLayer{T}"/>, <see cref="FullyConnectedLayer{T}"/>,
/// <see cref="FeedForwardLayer{T}"/>, <see cref="SparseLinearLayer{T}"/>,
/// <see cref="ConvolutionalLayer{T}"/>, <see cref="Conv3DLayer{T}"/>,
/// <see cref="DeconvolutionalLayer{T}"/>, <see cref="LocallyConnectedLayer{T}"/>,
/// <see cref="BidirectionalLayer{T}"/>, <see cref="LambdaLayer{T}"/>,
/// <see cref="MeshEdgeConvLayer{T}"/>, <see cref="SpiralConvLayer{T}"/>,
/// <see cref="TimeDistributedLayer{T}"/>. Default changed from <c>?? new ReLUActivation&lt;T&gt;()</c>
/// to <c>?? new IdentityActivation&lt;T&gt;()</c>. Matches Keras / PyTorch nn.Linear /
/// TF Dense conventions.</para>
///
/// <para>Two test families:</para>
/// <list type="number">
///   <item><b>Unit tests</b> — assert each layer's default <c>ScalarActivation</c> is
///   <see cref="IdentityActivation{T}"/>. Pre-fix these fail (default was ReLU).</item>
///   <item><b>Integration tests</b> — train a regression NN through the facade with
///   targets that span negative values, then assert the trained model produces
///   negative predictions for at least some samples. Pre-fix the ReLU output head
///   clamps every prediction to ≥ 0 and the test fails. Post-fix Identity passes
///   negative values through and the model learns the negative tail.</item>
/// </list>
/// </summary>
public class Issue1629_DenseLayerDefaultActivationTests
{
    private readonly ITestOutputHelper _output;
    public Issue1629_DenseLayerDefaultActivationTests(ITestOutputHelper output) => _output = output;

    // ── Unit tests: default activation is Identity, not ReLU ──

    [Fact]
    public void DenseLayer_DefaultActivation_IsIdentity()
        => AssertDefaultIsIdentity(new DenseLayer<double>(outputSize: 1));

    [Fact]
    public void DenseLayer_ExplicitNullActivation_IsIdentity()
        => AssertDefaultIsIdentity(new DenseLayer<double>(outputSize: 1, activationFunction: null));

    [Fact]
    public void DenseLayer_ExplicitReLU_StillIsReLU()
    {
        var layer = new DenseLayer<double>(outputSize: 1, activationFunction: new ReLUActivation<double>());
        Assert.IsType<ReLUActivation<double>>(layer.ScalarActivation);
    }

    [Fact]
    public void FullyConnectedLayer_DefaultActivation_IsIdentity()
        => AssertDefaultIsIdentity(new FullyConnectedLayer<double>(outputSize: 1));

    [Fact]
    public void FullyConnectedLayer_EagerCtor_DefaultActivation_IsIdentity()
        => AssertDefaultIsIdentity(new FullyConnectedLayer<double>(inputSize: 4, outputSize: 1));

    [Fact]
    public void FeedForwardLayer_DefaultActivation_IsIdentity()
        => AssertDefaultIsIdentity(new FeedForwardLayer<double>(outputSize: 1));

    [Fact]
    public void SparseLinearLayer_DefaultActivation_IsIdentity()
        => AssertDefaultIsIdentity(new SparseLinearLayer<double>(inputFeatures: 4, outputFeatures: 1));

    [Fact]
    public void ConvolutionalLayer_DefaultActivation_IsIdentity()
        => AssertDefaultIsIdentity(new ConvolutionalLayer<double>(outputDepth: 1, kernelSize: 3));

    [Fact]
    public void Conv3DLayer_DefaultActivation_IsIdentity()
        => AssertDefaultIsIdentity(new Conv3DLayer<double>(outputChannels: 1, kernelSize: 3));

    [Fact]
    public void DeconvolutionalLayer_DefaultActivation_IsIdentity()
        => AssertDefaultIsIdentity(new DeconvolutionalLayer<double>(outputDepth: 1, kernelSize: 3));

    [Fact]
    public void LocallyConnectedLayer_DefaultActivation_IsIdentity()
        => AssertDefaultIsIdentity(new LocallyConnectedLayer<double>(outputChannels: 1, kernelSize: 3, stride: 1));

    [Fact]
    public void LambdaLayer_DefaultActivation_IsIdentity()
    {
        // Disambiguate the scalar/vector activation overloads by passing typed null.
        var layer = new LambdaLayer<double>(
            inputShape: new[] { 4 },
            outputShape: new[] { 4 },
            forwardFunction: t => t,
            backwardFunction: null,
            activationFunction: (IActivationFunction<double>?)null);
        Assert.IsType<IdentityActivation<double>>(layer.ScalarActivation);
    }

    [Fact]
    public void MeshEdgeConvLayer_DefaultActivation_IsIdentity()
        => AssertDefaultIsIdentity(new MeshEdgeConvLayer<double>(
            inputChannels: 4, outputChannels: 1, numNeighbors: 3,
            activationFunction: (IActivationFunction<double>?)null));

    [Fact]
    public void SpiralConvLayer_DefaultActivation_IsIdentity()
        => AssertDefaultIsIdentity(new SpiralConvLayer<double>(outputChannels: 1, spiralLength: 3));

    [Fact]
    public void SpiralConvLayer_EagerCtor_DefaultActivation_IsIdentity()
        => AssertDefaultIsIdentity(new SpiralConvLayer<double>(inputChannels: 4, outputChannels: 1, spiralLength: 3));

    [Fact]
    public void TimeDistributedLayer_DefaultActivation_IsIdentity()
    {
        var inner = new DenseLayer<double>(outputSize: 4, activationFunction: new ReLUActivation<double>());
        var layer = new TimeDistributedLayer<double>(inner, activationFunction: (IActivationFunction<double>?)null);
        Assert.IsType<IdentityActivation<double>>(layer.ScalarActivation);
    }

    [Fact]
    public void BidirectionalLayer_DefaultActivation_IsIdentity()
    {
        var inner = new DenseLayer<double>(outputSize: 4, activationFunction: new ReLUActivation<double>());
        var layer = new BidirectionalLayer<double>(inner, activationFunction: (IActivationFunction<double>?)null);
        Assert.IsType<IdentityActivation<double>>(layer.ScalarActivation);
    }

    private static void AssertDefaultIsIdentity(LayerBase<double> layer)
    {
        Assert.IsType<IdentityActivation<double>>(layer.ScalarActivation);
    }

    // ── Integration tests: end-to-end facade build/train/predict ──

    /// <summary>
    /// Trains a small regression NN with DenseLayer as the output head, using targets
    /// that span negative values. Pre-fix: the silent ReLU default clamps every
    /// prediction to ≥ 0, so the model can never predict the negative tail — the
    /// "any prediction below 0" assertion fails. Post-fix: Identity allows negative
    /// output and the model learns to produce them.
    /// </summary>
    [Fact]
    public async System.Threading.Tasks.Task DenseLayer_AsRegressionOutput_CanProduceNegativePredictions()
    {
        await AssertRegressionWithDefaultOutputLayerProducesNegatives(
            outputLayer: new DenseLayer<double>(outputSize: 1));
    }

    [Fact]
    public async System.Threading.Tasks.Task FullyConnectedLayer_AsRegressionOutput_CanProduceNegativePredictions()
    {
        await AssertRegressionWithDefaultOutputLayerProducesNegatives(
            outputLayer: new FullyConnectedLayer<double>(outputSize: 1));
    }

    [Fact]
    public async System.Threading.Tasks.Task FeedForwardLayer_AsRegressionOutput_CanProduceNegativePredictions()
    {
        await AssertRegressionWithDefaultOutputLayerProducesNegatives(
            outputLayer: new FeedForwardLayer<double>(outputSize: 1));
    }

    /// <summary>
    /// Shared scenario: 200 train + 50 test samples, 4 features uniform in [-1, 1],
    /// target = sum of first 3 features (so range ~[-3, +3] with mean 0 and substantial
    /// negative tail). Pre-fix this fails for any layer whose default activation is
    /// ReLU because the model can never produce a negative output.
    /// </summary>
    private async System.Threading.Tasks.Task AssertRegressionWithDefaultOutputLayerProducesNegatives(
        LayerBase<double> outputLayer)
    {
        var (trainX, trainY, testX, testY) = MakeNegativeTargetData(trainN: 200, testN: 50, features: 4, seed: 42);

        // Verify the test data actually has negative targets, otherwise the test is meaningless.
        int negTrainCount = 0, negTestCount = 0;
        for (int i = 0; i < trainY.Shape[0]; i++) if (trainY[i, 0] < 0) negTrainCount++;
        for (int i = 0; i < testY.Shape[0]; i++) if (testY[i, 0] < 0) negTestCount++;
        Assert.True(negTrainCount > 50, $"Test data sanity: train targets must have substantial negative tail, got {negTrainCount}/200");
        Assert.True(negTestCount > 10, $"Test data sanity: test targets must have substantial negative tail, got {negTestCount}/50");

        // ReLU intermediate, default-activation output (the bug surface).
        var layers = new List<ILayer<double>>
        {
            new DenseLayer<double>(outputSize: 16, activationFunction: new ReLUActivation<double>()),
            new DenseLayer<double>(outputSize: 8,  activationFunction: new ReLUActivation<double>()),
            outputLayer,
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 4, outputSize: 1, layers: layers);
        var nn = new NeuralNetwork<double>(architecture);

        var builder = new AiModelBuilder<double, Tensor<double>, Tensor<double>>();
        var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(
            null, new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            { InitialLearningRate = 0.01, MaxIterations = 100, BatchSize = 32 });

        var model = await builder
            .ConfigureModel(nn)
            .ConfigureOptimizer(optimizer)
            .ConfigureLossFunction(new MeanSquaredErrorLoss<double>())
            .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(trainX, trainY))
            .BuildAsync();

        var preds = builder.Predict(testX, model);

        int n = preds.Shape[0];
        Assert.Equal(testX.Shape[0], n);

        double pMin = double.MaxValue, pMax = double.MinValue;
        int negPredCount = 0;
        for (int i = 0; i < n; i++)
        {
            double v = preds[i, 0];
            if (v < pMin) pMin = v;
            if (v > pMax) pMax = v;
            if (v < 0) negPredCount++;
        }
        _output.WriteLine($"{outputLayer.GetType().Name}: pred range=[{pMin:F4}, {pMax:F4}] · negative-pred count={negPredCount}/{n}");

        // The decisive assertion — pre-fix this is impossible because ReLU output is always ≥ 0.
        Assert.True(negPredCount > 0,
            $"Pre-fix bug reproduced: {outputLayer.GetType().Name} default-activation output never produced a negative " +
            $"prediction across {n} test samples (range [{pMin:F4}, {pMax:F4}]), even though the test set has " +
            $"{negTestCount} samples with negative true targets. With the silent ReLU default, the model literally " +
            "cannot learn to predict negative values — its output is clamped to ≥ 0. After the fix (default Identity), " +
            "the model can produce both positive and negative predictions.");
    }

    private static (Tensor<double> trainX, Tensor<double> trainY, Tensor<double> testX, Tensor<double> testY)
        MakeNegativeTargetData(int trainN, int testN, int features, int seed)
    {
        var (tx, ty) = MakeSplit(trainN, features, seed);
        var (ex, ey) = MakeSplit(testN, features, seed + 7);
        return (tx, ty, ex, ey);
    }

    private static (Tensor<double> X, Tensor<double> Y) MakeSplit(int n, int features, int seed)
    {
        // Features uniform in [-1, 1]. Target = sum of first 3 features ∈ [-3, +3].
        // Mean 0, substantial negative tail. The model MUST be able to output negative
        // values to fit this target distribution.
        var rng = new System.Random(seed);
        var X = new Tensor<double>(new[] { n, features });
        var Y = new Tensor<double>(new[] { n, 1 });
        for (int i = 0; i < n; i++)
        {
            double target = 0;
            for (int f = 0; f < features; f++)
            {
                var v = rng.NextDouble() * 2 - 1;
                X[i, f] = v;
                if (f < 3) target += v;
            }
            Y[i, 0] = target;
        }
        return (X, Y);
    }
}
