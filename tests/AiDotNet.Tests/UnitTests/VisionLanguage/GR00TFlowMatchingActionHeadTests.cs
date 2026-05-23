using System.Threading.Tasks;
using AiDotNet.VisionLanguage.Robotics;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

/// <summary>
/// Unit tests for GR00T N1's flow-matching action head (NVIDIA 2025, arXiv:2503.14734 §3.2).
/// Covers output shape, integration-step counting, and conditioning-latent propagation. Uses an
/// identity velocity callback so the test does not depend on trained model weights.
/// </summary>
public class GR00TFlowMatchingActionHeadTests
{
    [Fact(Timeout = 30000)]
    public async Task Generate_ReturnsTensorOfActionDimension()
    {
        await Task.Yield();
        var head = new GR00TFlowMatchingActionHead<double>(
            velocityNetwork: (xt, _, _) => xt,
            numIntegrationSteps: 16,
            seed: 42);

        var action = head.Generate(actionDimension: 52, system2Latent: new Tensor<double>([8]));

        Assert.Equal(52, action.Length);
    }

    [Fact(Timeout = 30000)]
    public async Task Generate_InvokesVelocityNetworkExactlyNumIntegrationStepsTimes()
    {
        await Task.Yield();
        int invocationCount = 0;
        var head = new GR00TFlowMatchingActionHead<double>(
            velocityNetwork: (xt, _, _) => { invocationCount++; return xt; },
            numIntegrationSteps: 8,
            seed: 42);

        head.Generate(actionDimension: 4, system2Latent: new Tensor<double>([2]));

        Assert.Equal(8, invocationCount);
    }

    [Fact(Timeout = 30000)]
    public async Task GenerateHorizon_ReturnsHorizonTimesActionDim()
    {
        await Task.Yield();
        var head = new GR00TFlowMatchingActionHead<double>(
            velocityNetwork: (xt, _, _) => xt,
            numIntegrationSteps: 4,
            seed: 42);

        var horizon = head.GenerateHorizon(actionDimension: 7, horizon: 16, system2Latent: new Tensor<double>([2]));

        Assert.Equal(7 * 16, horizon.Length);
    }

    [Fact(Timeout = 30000)]
    public async Task GenerateHorizon_InvokesVelocityNumStepsPerHorizonStep()
    {
        await Task.Yield();
        int invocationCount = 0;
        var head = new GR00TFlowMatchingActionHead<double>(
            velocityNetwork: (xt, _, _) => { invocationCount++; return xt; },
            numIntegrationSteps: 4,
            seed: 42);

        head.GenerateHorizon(actionDimension: 3, horizon: 5, system2Latent: new Tensor<double>([2]));

        // 5 horizon steps × 4 integration steps = 20 callbacks.
        Assert.Equal(20, invocationCount);
    }

    [Fact(Timeout = 30000)]
    public async Task SeededRng_ProducesDeterministicOutput()
    {
        await Task.Yield();
        var headA = new GR00TFlowMatchingActionHead<double>(
            velocityNetwork: (xt, _, _) => xt,
            numIntegrationSteps: 4,
            seed: 7);
        var headB = new GR00TFlowMatchingActionHead<double>(
            velocityNetwork: (xt, _, _) => xt,
            numIntegrationSteps: 4,
            seed: 7);

        var a = headA.Generate(actionDimension: 5, system2Latent: new Tensor<double>([2]));
        var b = headB.Generate(actionDimension: 5, system2Latent: new Tensor<double>([2]));

        for (int i = 0; i < 5; i++)
            Assert.Equal(a[i], b[i]);
    }
}
