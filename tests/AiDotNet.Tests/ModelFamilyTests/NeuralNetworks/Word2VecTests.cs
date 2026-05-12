using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class Word2VecTests : NeuralNetworkModelTestBase
{
    // Word2Vec's default ctor uses vocabSize=10000 — the last layer emits
    // a 10000-dim softmax over the vocabulary, so predicted output length
    // is 10000 per sample, not the [1, 1] implied by the base-class
    // default OutputShape. Align both sides so
    // OutputDimension_ShouldMatchExpectedShape compares like with like.
    protected override int[] InputShape => [1, 4];
    protected override int[] OutputShape => [1, 10000];

    // Paper-faithful index-based input per Mikolov et al. 2013 §3.1 —
    // Word2Vec's first layer is an EmbeddingLayer that expects integer
    // token IDs in [0, vocabSize). The base-class default emits doubles
    // in [0, 1) which cast to int 0 inside the embedding lookup, so
    // every "input" collapses to token 0 and only embedding[0] ever
    // receives a gradient. The remaining 9999 rows of the U matrix stay
    // frozen and the model can't memorize a 10000-class target, leaving
    // LossStrictlyDecreasesOnMemorizationTask saturating at ~0.6% loss
    // drop over 100 steps (vs the 1% threshold). The base-class
    // CreateRandomTensor's XML doc explicitly calls out this exact
    // override pattern; we just hadn't applied it.
    protected override Tensor<double> CreateRandomTensor(int[] shape, System.Random rng)
    {
        var tensor = new Tensor<double>(shape);
        // Word2Vec default vocabSize = 10000. Emit token indices in the
        // [0, 1000) sub-range so the test base's ScaledInput_ShouldChangeOutput
        // invariant (which multiplies the input by 10) still stays inside
        // the embedding's [0, vocabSize) bound after scaling. 1000 distinct
        // token IDs is plenty of input diversity for every other invariant
        // in the base (memorization, gradient flow, parameter change, etc.).
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = rng.Next(0, 1000);
        return tensor;
    }

    // Word2Vec's default loss is BinaryCrossEntropy, which expects targets
    // in [0, 1] (probabilities / one-hot encoding) — NOT the integer token
    // IDs that the input-side CreateRandomTensor emits. Override the target
    // factory to fall back to the base's default continuous [0, 1) sampler
    // so the target tensor stays compatible with BCE. Without this override,
    // the base's default CreateRandomTargetTensor delegates to our
    // CreateRandomTensor and the target gets out-of-range integer values
    // (0..999) that BCE then computes nonsensical log probabilities over.
    protected override Tensor<double> CreateRandomTargetTensor(int[] shape, System.Random rng)
    {
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = rng.NextDouble();
        return tensor;
    }

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new Word2Vec<double>();
}
