using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Model-family coverage for <see cref="ResidualNeuralNetwork{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// This is the one fixture in the family that runs in <c>double</c> rather than <c>float</c>, and the
/// reason is measurable rather than a preference. This network is a 30+ layer ReLU chain, and in
/// float32 its loss quantizes: with a base loss near 39.27, every finite-difference numerator
/// observed in <c>Gradients_MatchFiniteDifference</c> came out an exact integer multiple of
/// 5.976e-6 (observed multipliers 1, 2, 5, 8, 9, 15 and 26). The derivative being measured is around
/// 5.9e-4, so at the step sizes the ladder uses, the numerator is quantization granularity rather
/// than signal — the estimates swing between +8.6e-2 and −4.9e-4 and change sign four times across
/// the ladder.
/// </para>
/// <para>
/// The backward pass is fine. Running the identical check in <c>double</c> passes under the STRICTER
/// budget the base applies to double (allowed mismatches <c>max(1, checked/6)</c> = 2, against
/// <c>max(2, checked/3)</c> = 4 for float), and all 29 other tests in the family pass unchanged. So
/// this is not a loosened assertion: the gradient check here is now harder to satisfy than it was,
/// and it is measuring the gradient instead of float32 rounding.
/// </para>
/// </remarks>
public class ResidualNeuralNetworkTests : NeuralNetworkModelTestBase<double>
{
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [1];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new ResidualNeuralNetwork<double>();
}
