using System;
using System.Linq;
using System.Reflection;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Tabular;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public class Elementwise03bNumericalRegressionTests
{
    private const BindingFlags PrivateInstance = BindingFlags.Instance | BindingFlags.NonPublic;

    [Fact]
    public void ContrastiveLoss_ExcludesSelfBeforeExponentiatingAtLowTemperature()
    {
        var module = new ContrastivePretraining<double>(2, 2, projectionDim: 2, temperature: 0.001);
        var original = new Tensor<double>(new[] { 1.0, 0.0, -1.0, 0.0 }, new[] { 2, 2 });
        var corrupted = new Tensor<double>(new[] { 0.0, 1.0, 0.0, -1.0 }, new[] { 2, 2 });
        var method = module.GetType().GetMethod("ComputeInfoNCELoss", PrivateInstance)!;
        double loss = (double)method.Invoke(module, new object[] { original, corrupted, 2 })!;
        Assert.Equal(Math.Log(2), loss, 12);
    }

    [Fact]
    public void HalfMacroF1_LargeConfusionCountsMatchTheIntegerReference()
    {
        const int count = 120000;
        var predicted = new int[count];
        var targets = Enumerable.Range(0, count).Select(index => index < 90000 ? 0 : 1).ToArray();
        var method = typeof(FTTransformerClassifier<Half>).GetMethod("ComputeMacroF1",
            BindingFlags.Static | BindingFlags.NonPublic)!;
        var result = (Half)method.Invoke(null, new object[] { predicted, targets, 2 })!;
        // Class 0: 2*90000/(120000+90000)=6/7. Class 1: 0. Macro mean: 3/7.
        Assert.Equal((Half)(3.0 / 7), result);
    }

    [Fact]
    public void HalfGandalfTraining_WrongExtremeLogitsHaveFiniteLossAndGradients()
    {
        var options = new GANDALFOptions<Half>
        {
            NumTrees = 1, TreeDepth = 1, LeafDimension = 2,
            GatingHiddenDimension = 2, NumGatingLayers = 1,
            DropoutRate = 0, UseBatchNorm = false, InitScale = 0.1
        };
        var model = new GANDALFClassifier<Half>(2, 2, options);
        var head = (FullyConnectedLayer<Half>)model.GetType().GetField("_classificationHead", PrivateInstance)!.GetValue(model)!;
        var parameters = new Vector<Half>(checked((int)head.ParameterCount));
        Assert.Equal(6, parameters.Length);
        parameters[4] = (Half)30;
        parameters[5] = (Half)(-30);
        head.SetParameters(parameters);
        var input = new Tensor<Half>(new Half[] { (Half)1, (Half)0 }, new[] { 1, 2 });
        var loss = model.TrainStep(input, new Vector<int>(new[] { 1 }), (Half)0.1);
        Assert.InRange((double)loss, 59, 61);
        Assert.All(model.GetParameters().ToArray(), value => Assert.True(Half.IsFinite(value)));
        Assert.True(head.GetParameters()[4] < (Half)30);
        Assert.True(head.GetParameters()[5] > (Half)(-30));
    }

    [Fact]
    public void SubpixelSigmoidBackward_MatchesCentralDifferencesAtTheInput()
    {
        var layer = new SubpixelConvolutionalLayer<double>(1, 1, 1, activation: new SigmoidActivation<double>());
        var input = new Tensor<double>(new[] { 1.0, 0.5 }, new[] { 1, 1, 1, 2 });
        layer.Forward(input);
        layer.SetParameters(new Vector<double>(new[] { 1.0, 0.0 }));
        Tensor<double> actual;
        using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<double>())
        {
            var output = layer.Forward(input);
            var engine = AiDotNetEngine.Current;
            var upstream = new Tensor<double>(new[] { 1.0, 2.0 }, input.Shape.ToArray());
            var loss = engine.ReduceSum(engine.TensorMultiply(output, upstream), new[] { 0, 1, 2, 3 }, false);
            actual = tape.ComputeGradients(loss, new[] { input })[input];
        }
        const double step = 1e-6;
        for (int i = 0; i < input.Length; i++)
        {
            var plus = input.Clone();
            var minus = input.Clone();
            plus.SetFlat(i, plus.GetFlat(i) + step);
            minus.SetFlat(i, minus.GetFlat(i) - step);
            var high = layer.Forward(plus);
            double highLoss = high.GetFlat(0) + 2 * high.GetFlat(1);
            var low = layer.Forward(minus);
            double lowLoss = low.GetFlat(0) + 2 * low.GetFlat(1);
            Assert.InRange(Math.Abs(actual.GetFlat(i) - (highLoss - lowLoss) / (2 * step)), 0, 1e-8);
        }
    }

    [Fact]
    public void SubpixelSoftmaxBackward_IncludesOffDiagonalJacobianTerms()
    {
        var layer = new SubpixelConvolutionalLayer<double>(1, 1, 1,
            vectorActivation: new SoftmaxActivation<double>());
        var input = new Tensor<double>(new[] { 1.0, 0.5 }, new[] { 1, 1, 1, 2 });
        layer.Forward(input);
        layer.SetParameters(new Vector<double>(new[] { 1.0, 0.0 }));
        Tensor<double> gradient;
        using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<double>())
        {
            var output = layer.Forward(input);
            var engine = AiDotNetEngine.Current;
            var upstream = new Tensor<double>(new[] { 1.0, 2.0 }, input.Shape.ToArray());
            var loss = engine.ReduceSum(engine.TensorMultiply(output, upstream), new[] { 0, 1, 2, 3 }, false);
            gradient = tape.ComputeGradients(loss, new[] { input })[input];
        }
        double probability = 1 / (1 + Math.Exp(-0.5));
        double expected = probability * (1 - probability);
        Assert.Equal(-expected, gradient.GetFlat(0), 12);
        Assert.Equal(expected, gradient.GetFlat(1), 12);
    }

    [Fact]
    public void QuantumSubnormalInput_PreservesSignedUnitNormAmplitudes()
    {
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 2, outputSize: 1);
        var model = new QuantumNeuralNetwork<double>(architecture, numQubits: 1);
        var method = model.GetType().GetMethod("PrepareQuantumState", PrivateInstance)!;
        var state = (Tensor<Complex<double>>)method.Invoke(model, new object[]
        {
            new Tensor<double>(new[] { 1e-320, -1e-320 }, new[] { 2 })
        })!;
        Assert.Equal(1 / Math.Sqrt(2), state.GetFlat(0).Real, 12);
        Assert.Equal(-1 / Math.Sqrt(2), state.GetFlat(1).Real, 12);
        Assert.Equal(0, state.GetFlat(0).Imaginary);
    }

    private static MemoryNetwork<T> MemoryModel<T>() => new(
        new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 3, outputSize: 3), memorySize: 2, embeddingSize: 3);

    [Fact]
    public void MemoryMetadata_LargeFiniteFloatValuesHaveAFiniteMean()
    {
        var model = MemoryModel<float>();
        var memory = (Matrix<float>)model.GetType().GetField("_memory", PrivateInstance)!.GetValue(model)!;
        for (int row = 0; row < 2; row++)
            for (int column = 0; column < 3; column++) memory[row, column] = 1e38f;
        double average = (double)model.GetModelMetadata().AdditionalInfo["AvgMemoryValue"];
        Assert.Equal((double)1e38f, average);
    }

    [Fact]
    public void MemoryOverwrite_ReplacesNonFiniteValuesAndKeepsTheSuffix()
    {
        var model = MemoryModel<double>();
        var memory = (Matrix<double>)model.GetType().GetField("_memory", PrivateInstance)!.GetValue(model)!;
        memory[0, 0] = double.NaN;
        memory[0, 1] = double.PositiveInfinity;
        memory[0, 2] = 7;
        var method = model.GetType().GetMethod("WriteMemoryRow", PrivateInstance)!;
        method.Invoke(model, new object[] { 0,
            new Tensor<double>(new[] { 1.0, 2.0 }, new[] { 1, 2 }), 0.0, 1.0, 2 });
        Assert.Equal(new[] { 1.0, 2.0, 7.0 }, memory.GetRow(0).ToArray());
    }

    [Fact]
    public void SomQuantizationError_LargeFiniteFloatDistanceRemainsFinite()
    {
        var model = new SelfOrganizingMap<float>(new NeuralNetworkArchitecture<float>(InputType.OneDimensional,
            NeuralNetworkTaskType.Clustering, inputSize: 2, outputSize: 1));
        model.SetParameters(new Vector<float>(new float[] { 0, 0 }));
        model.Train(new Tensor<float>(new[] { 1e20f, 1e20f }, new[] { 2 }), new Tensor<float>(new[] { 1 }));
        Assert.True(float.IsFinite(model.GetLastLoss()));
        Assert.True(model.GetLastLoss() > 1e18f);
    }
}
