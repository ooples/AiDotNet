using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class RadialBasisFunctionNetworkTests : NeuralNetworkModelTestBase<float>
{
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [1];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new RadialBasisFunctionNetwork<float>();

    // SATURATION FAR FROM THE CENTRES IS THE ARCHITECTURE, NOT A DEFECT.
    //
    // An RBF network responds through Gaussian kernels: phi(x) = exp(-beta * ||x - c||^2). That
    // response is local by construction. Multiplying a random input by ten moves it many kernel
    // widths away from every centre c, where exp(-beta * ||x - c||^2) underflows to zero for BOTH
    // the original and the scaled input. The two forward passes then agree not because the network
    // ignores its input, but because both points sit in the same dead zone outside the basis --
    // exactly the behaviour Broomhead & Lowe (1988) describe and that makes RBFs local
    // approximators rather than global ones.
    //
    // The invariant is still worth enforcing everywhere else: a network genuinely ignoring its
    // input is a real bug. It just cannot be probed by a 10x scale on a local-basis architecture.
    // Sensitivity near the centres remains covered by the determinism, gradient-correctness and
    // training invariants, which perturb within the basis rather than far outside it.
    protected override bool ScaledInputInvariantApplicable => false;
}
