using System.Reflection;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Helpers;

/// <summary>
/// Checks that a classifier's head and its loss agree on what the head emits: logits for a loss that applies its own
/// softmax (<see cref="CrossEntropyWithLogitsLoss{T}"/>), probabilities for a loss that consumes them (one declared
/// <see cref="LossPropertyAttribute.RequiresProbabilityInputs"/>, such as <see cref="CategoricalCrossEntropyLoss{T}"/>).
/// </summary>
/// <remarks>
/// Either mismatch trains on the wrong quantity without an error: a softmax head under the logits loss normalizes twice
/// (the gradients flatten toward uniform), and a logits head under a probability loss takes the log of raw scores
/// (five zero logits and a one-hot target give -log(1e-7) instead of log 5).
/// </remarks>
internal static class ClassificationHeadContract
{
    /// <summary>Whether the layer's activation is a softmax, so its output is already a class distribution.</summary>
    public static bool AppliesSoftmax<T>(ILayer<T> layer)
        => layer is LayerBase<T> withActivation
            && ((object?)withActivation.VectorActivation ?? withActivation.ScalarActivation) is SoftmaxActivation<T>;

    /// <summary>Whether the loss declares that its predictions must be probabilities.</summary>
    public static bool RequiresProbabilities<T>(ILossFunction<T> loss)
        => loss.GetType().GetCustomAttribute<LossPropertyAttribute>(inherit: true) is { RequiresProbabilityInputs: true };

    /// <summary>
    /// Throws when the head's output kind and the loss disagree.
    /// </summary>
    /// <param name="headEmitsProbabilities">Whether the head ends in a softmax.</param>
    /// <param name="loss">The training loss.</param>
    /// <param name="parameterName">The constructor parameter the exception names.</param>
    public static void Validate<T>(bool headEmitsProbabilities, ILossFunction<T> loss, string parameterName)
    {
        if (headEmitsProbabilities && loss is CrossEntropyWithLogitsLoss<T>)
            throw new ArgumentException(
                "The classification head applies a softmax, but CrossEntropyWithLogitsLoss applies its own. Give the head "
                + "an identity activation (logits), or use a probability loss such as CategoricalCrossEntropyLoss.",
                parameterName);
        if (!headEmitsProbabilities && RequiresProbabilities(loss))
            throw new ArgumentException(
                $"The classification head emits logits, but {loss.GetType().Name} requires probabilities. Use "
                + "CrossEntropyWithLogitsLoss, or end the head with a softmax activation.",
                parameterName);
    }
}
