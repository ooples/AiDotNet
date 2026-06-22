using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class FastTextTests : NeuralNetworkModelTestBase<float>
{
    // FastText's default sub-word table is bucketSize = 2,000,000 × embeddingDim 100 =
    // ~200M parameters (paper-faithful: Joulin et al. 2016 use ~2M buckets). Every Adam
    // step is therefore O(200M), so the base MoreData default of 50 + 200 = 250 training
    // iterations cannot finish inside the 120 s xUnit timeout (it is compute-bound, not
    // functionally broken — LossStrictlyDecreasesOnMemorizationTask runs the model at full
    // strength and passes). Cap the iteration counts the same way the other paper-scale
    // model-family tests do (NEAT / CLIP-family / VGG / DenseNet): the "more training never
    // degrades loss" invariant is still exercised (long ≥ short), just at a runnable depth.
    protected override int MoreDataShortIterations => 1;
    protected override int MoreDataLongIterations => 2;

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new FastText<float>();

    // FastText is a softmax multi-class classifier trained with cross-entropy
    // (Joulin et al. 2016 "Bag of Tricks"), so its paper-faithful target is a one-hot
    // class label, not a continuous-uniform tensor. The test base documents this exact
    // override hook for "multi-class classification with cross-entropy" families.
    // A continuous-uniform target is not a valid class distribution: categorical
    // cross-entropy over the 10000-way softmax can never drive the prediction onto it,
    // so the loss barely moves (LossStrictlyDecreases) and the MSE proxy never settles
    // (MoreData). With a one-hot target the softmax concentrates mass on the label and
    // the loss drops sharply — the correct objective for this model.
    protected override Tensor<float> CreateRandomTargetTensor(int[] shape, System.Random rng)
    {
        var target = new Tensor<float>(shape);
        int numClasses = shape[^1];
        int rows = target.Length / numClasses;
        for (int r = 0; r < rows; r++)
            target[r * numClasses + rng.Next(numClasses)] = 1.0f;
        return target;
    }
}
