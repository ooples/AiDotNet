using System;
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
/// objective phases) feed it the same public input that Predict takes. These four models own their Train step, so
/// the census stopped probing their training forward (#2414); the forward itself still walked the composite
/// <c>Layers</c> list as one chain and threw on every public input (#2387). Each case builds the model exactly as its
/// generated model-family fixture does, then requires the training forward to accept the fixture's input, to keep
/// Predict's output shape, and to reach the model's parameters on the tape.
/// </summary>
public class PublicTrainingForwardTests
{
    [Theory(Timeout = 120000)]
    [InlineData("WGANGP")]
    [InlineData("GOTOCR2")]
    [InlineData("CLAPModel")]
    [InlineData("OpenSora")]
    public async Task ForwardForTraining_AcceptsThePublicInputAndReachesParameters(string model)
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
            int gradientCount;
            int[] trained;
            using (var tape = new GradientTape<float>())
            {
                var output = network.ForwardForTraining(input);
                trained = output.Shape.ToArray();
                var objective = AiDotNetEngine.Current.ReduceSum(AiDotNetEngine.Current.TensorMultiply(output, output), null);
                var parameters = TapeTrainingStep<float>.CollectParameters(network.Layers, structureVersion: -1);
                var gradients = tape.ComputeGradients(objective, parameters);
                gradientCount = gradients.Count(kv => kv.Value is not null);
            }

            Assert.Equal(predicted, trained);
            Assert.True(gradientCount > 0, $"{model}: the training forward reached no parameter on the tape.");
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
