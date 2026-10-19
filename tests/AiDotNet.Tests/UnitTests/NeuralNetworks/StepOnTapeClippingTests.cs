using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// A joint step over several networks clips each network's gradient by its own norm (PyTorch's per-optimizer
/// clip_grad_norm_). Each optimizer used to receive every member's gradients: its global-norm clip measured and
/// scaled all of them, so each network was clipped by the joint norm and the next member clipped again on gradients
/// the previous one had already rescaled.
/// </summary>
public class StepOnTapeClippingTests
{
    private static FeedForwardNeuralNetwork<double> Network(int input, int output) => new(
        new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression, inputSize: input, outputSize: output,
            layers: new List<ILayer<double>>
            {
                new DenseLayer<double>(output, (IActivationFunction<double>)new IdentityActivation<double>()),
            }));

    private static AdamOptimizer<double, Tensor<double>, Tensor<double>> ClippingAdam(NeuralNetworkBase<double> network) => new(
        network, new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            InitialLearningRate = 0.01, EnableGradientClipping = true, MaxGradientNorm = 0.5,
        });

    private static void Fill(NeuralNetworkBase<double> network, int seed)
    {
        var parameters = network.GetParameters();
        for (int i = 0; i < parameters.Length; i++) parameters[i] = 0.4 * Math.Sin(0.7 * i + seed);
        network.SetParameters(parameters);
    }

    // loss = sum((B(A(x)))^2) * 10: large gradients on both networks, so every clip engages.
    private static Tensor<double> Loss(NeuralNetworkBase<double> a, NeuralNetworkBase<double> b, Tensor<double> x)
    {
        var engine = AiDotNetEngine.Current;
        var y = b.ForwardForTraining(a.ForwardForTraining(x));
        return engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorMultiply(y, y), new[] { 0, 1 }, keepDims: false), 10.0);
    }

    private enum Stepped { Both, FirstOnly, SecondOnly }

    [Fact]
    public void JointStep_ClipsEachNetworkByItsOwnNorm()
    {
        var x = new Tensor<double>([2, 3]);
        for (int i = 0; i < x.Length; i++) x[i] = 1.0 + 0.5 * i;

        (Vector<double> A, Vector<double> B) Run(Stepped mode)
        {
            var a = Network(3, 4);
            var b = Network(4, 2);
            a.Predict(x);
            b.Predict(new Tensor<double>([2, 4]));
            Fill(a, 1);
            Fill(b, 2);
            var members = new List<(NeuralNetworkBase<double>, IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>?)>();
            if (mode != Stepped.SecondOnly) members.Add((a, ClippingAdam(a)));
            if (mode != Stepped.FirstOnly) members.Add((b, ClippingAdam(b)));
            using (var tape = new GradientTape<double>())
                NeuralNetworkBase<double>.StepOnTape(tape, Loss(a, b, x), members);
            return (a.GetParameters(), b.GetParameters());
        }

        var joint = Run(Stepped.Both);
        var firstAlone = Run(Stepped.FirstOnly).A;
        var secondAlone = Run(Stepped.SecondOnly).B;
        for (int i = 0; i < joint.A.Length; i++)
            Assert.True(Math.Abs(joint.A[i] - firstAlone[i]) <= 1e-12, $"A weight {i}: joint {joint.A[i]}, alone {firstAlone[i]}");
        for (int i = 0; i < joint.B.Length; i++)
            Assert.True(Math.Abs(joint.B[i] - secondAlone[i]) <= 1e-12, $"B weight {i}: joint {joint.B[i]}, alone {secondAlone[i]}");
    }
}