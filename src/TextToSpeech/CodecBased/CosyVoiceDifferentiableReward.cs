using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>A differentiable token-domain reward for CosyVoice 3 DiffRO (paper equations 3-7).</summary>
/// <remarks>The reward's weights remain frozen during policy training, but its input derivatives must be retained.
/// Implementations return a scalar log reward, not a detached numeric score or a waveform reward.</remarks>
public interface ICosyVoiceTokenReward<T>
{
    /// <summary>The number of acoustic speech-code categories accepted along the last input axis.</summary>
    int SpeechVocabularySize { get; }

    /// <summary>The reward's layers, put in evaluation mode during policy optimization.</summary>
    IReadOnlyList<LayerBase<T>> Layers { get; }

    /// <summary>Returns the scalar log reward of soft speech tokens [frames, categories] for the target transcript.</summary>
    Tensor<T> LogReward(Tensor<T> softSpeechTokens, IReadOnlyList<int> textTokens);
}

/// <summary>Explicit numerical choices for DiffRO; the paper does not publish their fitted values.</summary>
public sealed class CosyVoiceDiffROOptions
{
    /// <summary>Coefficient beta of the summed token-level forward KL, equation 5.</summary>
    public double KlCoefficient { get; set; } = 0.1;

    /// <summary>Temperature of the differentiable, soft Gumbel-Softmax sample.</summary>
    public double Temperature { get; set; } = 1;

    internal void Validate()
    {
        if (double.IsNaN(KlCoefficient) || double.IsInfinity(KlCoefficient) || KlCoefficient < 0)
            throw new ArgumentOutOfRangeException(nameof(KlCoefficient));
        if (double.IsNaN(Temperature) || double.IsInfinity(Temperature) || Temperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(Temperature));
    }
}

internal static class CosyVoiceDiffRO<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    internal static (Tensor<T> Loss, Tensor<T> SoftTokens, Tensor<T> Kl, Tensor<T> Reward) Objective(
        IEngine engine, Tensor<T> policyLogits, Tensor<T> referenceLogits, Func<Tensor<T>, Tensor<T>> logReward,
        CosyVoiceDiffROOptions options, Random random)
    {
        options.Validate();
        if (policyLogits.Rank != 2 || policyLogits.Shape[0] == 0 || policyLogits.Shape[1] < 2
            || !policyLogits.Shape.ToArray().SequenceEqual(referenceLogits.Shape.ToArray()))
            throw new ArgumentException("DiffRO requires matching nonempty [frames, speech categories] logits.");
        // The reference posterior is a constant even if a caller computed it on a tape.
        var frozen = new Tensor<T>(referenceLogits.Shape.ToArray(), new Vector<T>(referenceLogits.ToArray()));
        var logPolicy = engine.TensorLogSoftmax(policyLogits, axis: 1);
        var logReference = engine.TensorLogSoftmax(frozen, axis: 1);
        var kl = engine.ReduceSum(engine.TensorMultiply(engine.TensorExp(logPolicy),
            engine.TensorSubtract(logPolicy, logReference)), new[] { 0, 1 }, false);
        var noise = new Tensor<T>(policyLogits.Shape.ToArray());
        for (int i = 0; i < noise.Length; i++)
        {
            double uniform = MathPolyfill.Clamp(random.NextDouble(), 1e-12, 1 - 1e-12);
            noise.SetFlat(i, Ops.FromDouble(-Math.Log(-Math.Log(uniform))));
        }
        var soft = engine.Softmax(engine.TensorMultiplyScalar(engine.TensorAdd(policyLogits, noise),
            Ops.FromDouble(1 / options.Temperature)), axis: 1);
        var reward = logReward(soft);
        if (reward.Length != 1) throw new ArgumentException("The differentiable log reward must be scalar.", nameof(logReward));
        var loss = engine.TensorSubtract(engine.TensorMultiplyScalar(kl, Ops.FromDouble(options.KlCoefficient)),
            engine.Reshape(reward, kl.Shape.ToArray()));
        return (loss, soft, kl, reward);
    }
}
