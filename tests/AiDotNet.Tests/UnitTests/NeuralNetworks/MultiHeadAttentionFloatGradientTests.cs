using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public class MultiHeadAttentionFloatGradientTests
{
    [Fact(Timeout = 30000)]
    public async Task ValueProjection_GradientMatchesFiniteDifferenceAtSmokeModelWidth()
    {
        await Task.Yield();

        var layer = new MultiHeadAttentionLayer<float>(4, 16);
        layer.SetTrainingMode(false);
        var input = new Tensor<float>(new[] { 1, 4, 64 });
        for (int i = 0; i < input.Length; i++)
            input[i] = (float)(0.15 * Math.Sin(i * 0.17) + 0.05 * Math.Cos(i * 0.07));

        var target = new Tensor<float>(new[] { 1, 4, 64 });
        for (int i = 0; i < target.Length; i++)
            target[i] = (float)(0.1 * Math.Cos(i * 0.11));

        var loss = new MeanSquaredErrorLoss<float>();
        using var tape = new GradientTape<float>();
        var objective = loss.ComputeTapeLoss(layer.Forward(input), target);
        var parameters = layer.GetTrainableParameters();
        var gradients = tape.ComputeGradients(objective, parameters);

        Assert.True(parameters.Count >= 3, "Multi-head attention must expose Q, K, and V projections.");
        var valueWeights = parameters[2];
        Assert.True(gradients.TryGetValue(valueWeights, out var analytical));

        const int index = 54;
        const float epsilon = 5e-3f;
        float original = valueWeights[index];
        valueWeights[index] = original + epsilon;
        float plus = loss.ComputeTapeLoss(layer.Forward(input), target)[0];
        valueWeights[index] = original - epsilon;
        float minus = loss.ComputeTapeLoss(layer.Forward(input), target)[0];
        valueWeights[index] = original;

        double numerical = (plus - minus) / (2.0 * epsilon);
        double expected = analytical[index];
        double relativeError = Math.Abs(expected - numerical) /
            Math.Max(1e-3, Math.Abs(expected) + Math.Abs(numerical));
        Assert.True(relativeError < 0.05,
            $"Value-projection gradient differs at index {index}: analytical={expected:E6}, " +
            $"numerical={numerical:E6}, relative error={relativeError:F4}.");
    }

    [Fact(Timeout = 30000)]
    public async Task CrossAttention_ContextGradientMatchesFiniteDifferenceInEvaluationMode()
    {
        await Task.Yield();

        var layer = new MultiHeadAttentionLayer<float>(2, 8);
        layer.SetTrainingMode(false);

        var query = new Tensor<float>(new[] { 1, 2, 16 });
        for (int i = 0; i < query.Length; i++)
            query[i] = (float)(0.12 * Math.Sin(i * 0.19) - 0.03 * Math.Cos(i * 0.07));

        var context = new Tensor<float>(new[] { 1, 3, 16 });
        for (int i = 0; i < context.Length; i++)
            context[i] = (float)(0.09 * Math.Cos(i * 0.13) + 0.04 * Math.Sin(i * 0.23));

        var target = new Tensor<float>(new[] { 1, 2, 16 });
        for (int i = 0; i < target.Length; i++)
            target[i] = (float)(0.08 * Math.Sin(i * 0.17));

        var loss = new MeanSquaredErrorLoss<float>();
        using var tape = new GradientTape<float>();
        var objective = loss.ComputeTapeLoss(layer.Forward(query, context), target);
        var gradients = tape.ComputeGradients(objective, new[] { context });

        Assert.True(gradients.TryGetValue(context, out var analytical),
            "Evaluation-mode cross-attention must preserve the tape path to its context input.");

        const int index = 19;
        const float epsilon = 2e-3f;
        float original = context[index];
        context[index] = original + epsilon;
        float plus = loss.ComputeTapeLoss(layer.Forward(query, context), target)[0];
        context[index] = original - epsilon;
        float minus = loss.ComputeTapeLoss(layer.Forward(query, context), target)[0];
        context[index] = original;

        double numerical = (plus - minus) / (2.0 * epsilon);
        double expected = analytical[index];
        double relativeError = Math.Abs(expected - numerical) /
            Math.Max(1e-3, Math.Abs(expected) + Math.Abs(numerical));
        Assert.True(relativeError < 0.05,
            $"Cross-attention context gradient differs at index {index}: analytical={expected:E6}, " +
            $"numerical={numerical:E6}, relative error={relativeError:F4}.");
    }
}
