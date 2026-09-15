using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class DeepBeliefNetworkTests : NeuralNetworkModelTestBase<float>
{
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [1];

    // Pin construction to a fixed seed so the fixture is reproducible. The default architecture carries
    // no RandomSeed, so each RBM's Glorot init came from CreateSecureRandom and every run trained a
    // different network. With the seed in scope, the RBM layers also draw their contrastive-divergence
    // Gibbs samples from that seeded stream (RBMLayer.SampleBinaryStatesTensor), so PreTrain and the
    // supervised fine-tune are deterministic end to end. AmbientFallbackSeed is [ThreadStatic], so
    // this is parallel-safe; same pattern as SpikingNeuralNetworkTests and SpiralNetTests.
    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        var previousSeed = AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.AmbientFallbackSeed;
        AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.AmbientFallbackSeed = 1337;
        try { return new DeepBeliefNetwork<float>(); }
        finally { AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.AmbientFallbackSeed = previousSeed; }
    }

    // Per Hinton 2006 ("A fast learning algorithm for deep belief nets") and
    // Hinton & Salakhutdinov 2006 ("Reducing the Dimensionality of Data with
    // Neural Networks"), a DBN's supervised-training contract starts with
    // greedy layer-wise CD-1 pretraining. Keep that model-specific phase here;
    // the shared base owns the invariant mechanics, objective measurement,
    // iteration policy, and diagnostics.
    protected override void PrepareForSupervisedTrainingInvariant(
        INeuralNetworkModel<float> network,
        Tensor<float> input)
        => ((DeepBeliefNetwork<float>)network).PreTrain(input);

    // CD-1 is stochastic by design. Near a converged floor, reconstruction
    // noise can be larger than the smooth-gradient default tolerance while the
    // supervised objective remains stable and finite.
    protected override double TrainingLossReductionTolerance => 5e-3;
}
