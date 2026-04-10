using AiDotNet.HarmonicEngine.Enums;
using AiDotNet.HarmonicEngine.Models;
using AiDotNet.HarmonicEngine.Options;
using AiDotNet.Tensors;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.HarmonicEngine;

public class HREModelSpectralGatingTests : HREModelTestBase
{
    protected override HREModel<double> CreateModel()
        => new HREModel<double>(new HREModelOptions
        {
            InputSize = 64, OutputSize = 1, CarrierCount = 8, FftSize = 256,
            Nonlinearity = NonlinearityType.SpectralGating,
            UseMellinFourier = false, NumOFDMLayers = 1, NumAttentionLayers = 1,
            HebbianLearningRate = 0.01, Seed = 42
        });
}

public class HREModelModReLUTests : HREModelTestBase
{
    protected override HREModel<double> CreateModel()
        => new HREModel<double>(new HREModelOptions
        {
            InputSize = 64, OutputSize = 1, CarrierCount = 8, FftSize = 256,
            Nonlinearity = NonlinearityType.ModReLU,
            UseMellinFourier = false, NumOFDMLayers = 1, NumAttentionLayers = 0,
            HebbianLearningRate = 0.01, Seed = 42
        });
}

public class HREModelInstantaneousFreqTests : HREModelTestBase
{
    protected override HREModel<double> CreateModel()
        => new HREModel<double>(new HREModelOptions
        {
            InputSize = 64, OutputSize = 1, CarrierCount = 8, FftSize = 256,
            Nonlinearity = NonlinearityType.InstantaneousFreq,
            UseMellinFourier = false, NumOFDMLayers = 1, NumAttentionLayers = 1,
            HebbianLearningRate = 0.01, Seed = 42
        });
}

public class HREModelWithMellinFourierTests : HREModelTestBase
{
    protected override HREModel<double> CreateModel()
        => new HREModel<double>(new HREModelOptions
        {
            InputSize = 64, OutputSize = 1, CarrierCount = 8, FftSize = 256,
            Nonlinearity = NonlinearityType.SpectralGating,
            UseMellinFourier = true, NumOFDMLayers = 1, NumAttentionLayers = 0,
            HebbianLearningRate = 0.01, Seed = 42
        });
}

public class HREModelMultiOutputTests : HREModelTestBase
{
    protected override int[] OutputShape => [4];

    protected override HREModel<double> CreateModel()
        => new HREModel<double>(new HREModelOptions
        {
            InputSize = 64, OutputSize = 4, CarrierCount = 8, FftSize = 256,
            Nonlinearity = NonlinearityType.SpectralGating,
            UseMellinFourier = false, NumOFDMLayers = 1, NumAttentionLayers = 0,
            HebbianLearningRate = 0.01, Seed = 42
        });
}

public class HREModelDeepPipelineTests : HREModelTestBase
{
    protected override HREModel<double> CreateModel()
        => new HREModel<double>(new HREModelOptions
        {
            InputSize = 64, OutputSize = 1, CarrierCount = 8, FftSize = 256,
            Nonlinearity = NonlinearityType.SpectralGating,
            UseMellinFourier = false, NumOFDMLayers = 3, NumAttentionLayers = 1,
            HebbianLearningRate = 0.01, Seed = 42
        });
}
