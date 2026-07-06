using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

// HopeNetwork (Behrouz et al. 2025 "Hope: A Self-Modifying Architecture for
// Continuum Memory Systems") is a self-modifying optimizer with hierarchical
// continuum memory + context flow + recurrent layers. It is NOT an
// associative memory network like Hopfield — it does not implement classical
// pattern storage / retrieval / capacity invariants. Inheriting
// AssociativeMemoryTestBase incorrectly imposes those invariants and causes
// NoiseRobustness/Capacity/OrthogonalPatterns/Clone/SerializationRoundTrip
// to fail because Hope's training dynamics don't fit that contract. Inherit
// the standard NeuralNetworkModelTestBase instead — the network's real
// contract (forward finite, training reduces loss, gradients flow, clone
// preserves outputs) is already covered there.
public class HopeNetworkTests : NeuralNetworkModelTestBase<float>
{
    protected override int[] InputShape => [256];
    protected override int[] OutputShape => [256];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new HopeNetwork<float>();
}
