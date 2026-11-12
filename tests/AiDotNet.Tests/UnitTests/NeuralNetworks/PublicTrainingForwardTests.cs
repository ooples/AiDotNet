using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// ForwardForTraining is public, and the tape-based paths (BuildTrainingObjective, gradient checks, the census'
/// objective phases) feed it the same public input that Predict takes. These three models own their Train step, so
/// the census stopped probing their training forward (#2414); the forward itself still walked the composite
/// <c>Layers</c> list as one chain and threw on every public input (#2387). Each case builds the model exactly as its
/// generated model-family fixture does, then requires the training forward to accept the fixture's input, to keep
/// Predict's output shape, and to give a non-zero gradient to every layer with parameters except the ones the
/// training forward does not run, which each case names (measured: the layers listed there get exactly zero).
/// </summary>
public class PublicTrainingForwardTests
{
    [Theory(Timeout = 120000)]
    // WGAN-GP: layers 7, 8 and 10 are the second network's convolutions and head, which this forward does not run.
    [InlineData("WGANGP", new[] { 7, 8, 10 })]
    [InlineData("GOTOCR2", new int[0])]
    [InlineData("CLAPModel", new int[0])]
    public async Task ForwardForTraining_AcceptsThePublicInputAndReachesParameters(string model, int[] notRun)
    {
        await Task.Yield();

        var fixtureType = typeof(PublicTrainingForwardTests).Assembly.GetType(
            $"AiDotNet.Tests.ModelFamilyTests.Generated.{model}Tests", throwOnError: true)!;
        object fixture = Activator.CreateInstance(fixtureType)!;
        var network = (NeuralNetworkBase<float>)Invoke(fixture, "CreateNetwork");
        try
        {
            var shape = (int[])Property(fixture, "EffectiveInputShape");
            var input = (Tensor<float>)Invoke(fixture, "CreateRandomTensor", shape, new Random(42));

            network.SetTrainingMode(false);
            int[] predicted = network.Predict(input).Shape.ToArray();

            network.SetTrainingMode(true);
            int[] trained;
            var unreached = new List<int>();
            using (var tape = new GradientTape<float>())
            {
                var output = network.ForwardForTraining(input);
                trained = output.Shape.ToArray();
                var objective = AiDotNetEngine.Current.ReduceSum(AiDotNetEngine.Current.TensorMultiply(output, output), null);
                var parameters = TapeTrainingStep<float>.CollectParameters(network.Layers, structureVersion: -1);
                var gradients = tape.ComputeGradients(objective, parameters);
                for (int i = 0; i < network.Layers.Count; i++)
                {
                    var own = TapeTrainingStep<float>.CollectParameters(new[] { network.Layers[i] }, structureVersion: -1);
                    if (own.Count == 0) continue;
                    bool reached = own.Any(t => gradients.TryGetValue(t, out var g) && g is not null
                        && g.ToArray().Any(v => v != 0 && !float.IsNaN(v) && !float.IsInfinity(v)));
                    if (!reached) unreached.Add(i);
                }
            }

            Assert.Equal(predicted, trained);
            Assert.True(notRun.SequenceEqual(unreached),
                $"{model}: layers without a non-zero gradient [{string.Join(", ", unreached)}], expected [{string.Join(", ", notRun)}] "
                + $"({string.Join(", ", unreached.Select(i => network.Layers[i].GetType().Name))}).");
        }
        finally
        {
            (network as IDisposable)?.Dispose();
        }
    }

    private static object Invoke(object target, string name, params object[] args)
    {
        for (var type = target.GetType(); type is not null; type = type.BaseType)
        {
            var method = type.GetMethod(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly);
            if (method is not null) return method.Invoke(target, args)!;
        }
        throw new MissingMethodException(target.GetType().Name, name);
    }

    private static object Property(object target, string name)
    {
        for (var type = target.GetType(); type is not null; type = type.BaseType)
        {
            var property = type.GetProperty(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly);
            if (property is not null) return property.GetValue(target)!;
        }
        throw new MissingMemberException(target.GetType().Name, name);
    }
}
