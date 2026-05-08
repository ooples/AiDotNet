using AiDotNet.Interfaces;
using AiDotNet.Safety.Adversarial;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for AdversarialImageEvaluator. Per Xu et al. 2018
/// (Feature Squeezing) the detector takes an NCHW image and produces a per-image
/// adversarial-detection score in [0, 1] via a learnable weighted ensemble of
/// three heuristic features (HF energy, histogram gaps, feature-squeezing
/// residual). The auto-generator can't construct it (parameterless ctor with
/// default-arg threshold isn't recognised); this manual class supplies the
/// ctor explicitly.
/// </summary>
public class AdversarialImageEvaluatorTests : NeuralNetworkModelTestBase
{
    protected override int[] InputShape => [1, 3, 32, 32];
    protected override int[] OutputShape => [1, 1];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new AdversarialImageEvaluator<double>(threshold: 0.5);
}
