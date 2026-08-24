using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class DeepBeliefNetworkTests : NeuralNetworkModelTestBase<float>
{
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [1];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new DeepBeliefNetwork<float>();

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
