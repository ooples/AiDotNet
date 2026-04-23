using AiDotNet.Enums;
using AiDotNet.Finance.Forecasting.Foundation;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Forecasting;

/// <summary>
/// Manual factory for <see cref="MGTSD{T}"/>. The auto-generator emits a
/// <c>NotImplementedException</c> placeholder because the constructor exposes
/// two overloads (an ONNX path and a native path) and the auto-detector can't
/// disambiguate without manual help.
/// </summary>
/// <remarks>
/// Per Shen et al. 2024 ("MG-TSD: Multi-Granularity Time Series Diffusion
/// Models"), MGTSD trains a diffusion model over multiple temporal grains.
/// The smoke shape uses the default option values (ContextLength=168,
/// ForecastHorizon=24) so the test invariants exercise the paper's standard
/// configuration.
/// </remarks>
public class MGTSDTests : NeuralNetworkModelTestBase
{
    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 168,
            outputSize: 24);
        return new MGTSD<double>(arch);
    }

    protected override int[] InputShape => new[] { 168 };
    protected override int[] OutputShape => new[] { 24 };
}
