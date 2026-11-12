using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Gaussian IPA-prior KL through OpenVoice's volume-preserving mean-only flow.
/// This is the VITS KL estimator used with the alignment permitted by paper section 2.3.</summary>
internal static class OpenVoiceTrainingLoss<T>
{
    internal static Tensor<T> AlignedKl(IEngine engine, Tensor<T> transformedPosterior, Tensor<T> posteriorLogScale,
        Tensor<T> alignedMean, Tensor<T> alignedLogScale, Tensor<T> mask)
    {
        if (transformedPosterior.Rank != 3 || !transformedPosterior.Shape.Equals(posteriorLogScale.Shape) ||
            !transformedPosterior.Shape.Equals(alignedMean.Shape) || !transformedPosterior.Shape.Equals(alignedLogScale.Shape) ||
            mask.Rank != 3 || mask.Shape[0] != transformedPosterior.Shape[0] || mask.Shape[1] != 1 || mask.Shape[2] != transformedPosterior.Shape[2])
            throw new ArgumentException("Expected matching [batch,channels,frames] Gaussian tensors and a [batch,1,frames] mask.");
        var numeric = MathHelper.GetNumericOperations<T>();
        double frames = mask.ToArray().Sum(x => numeric.ToDouble(x));
        if (frames <= 0) throw new ArgumentException("A nonempty alignment mask is required.", nameof(mask));
        var difference = engine.TensorSubtract(transformedPosterior, alignedMean);
        var inverseVariance = engine.TensorExp(engine.TensorMultiplyScalar(alignedLogScale, numeric.FromDouble(-2)));
        var terms = engine.TensorAdd(engine.TensorAddScalar(engine.TensorSubtract(alignedLogScale, posteriorLogScale), numeric.FromDouble(-.5)),
            engine.TensorMultiplyScalar(engine.TensorMultiply(engine.TensorMultiply(difference, difference), inverseVariance), numeric.FromDouble(.5)));
        terms = engine.TensorMultiply(terms, engine.TensorTile(mask, new[] { 1, transformedPosterior.Shape[1], 1 }));
        return engine.Reshape(engine.TensorMultiplyScalar(engine.ReduceSum(terms, new[] { 0, 1, 2 }, true), numeric.FromDouble(1 / frames)), new[] { 1 });
    }
}
