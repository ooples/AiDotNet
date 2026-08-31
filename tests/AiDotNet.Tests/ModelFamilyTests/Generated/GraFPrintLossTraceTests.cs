using AiDotNet.Audio.Fingerprinting;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Generated;

public class GraFPrintLossTraceTests : EmbeddingModelTestBase<float>
{
    // Bounded widths/depth keep the conformance suite fast while retaining every
    // paper operation: coordinate peak extraction, dynamic max-relative graph
    // convolution, both residual paths, and the SimCLR projector.
    private const int Batch = 4;
    protected override int[] InputShape => new[] { Batch, 1, 16, 16 };
    protected override int[] OutputShape => new[] { Batch, 4 };

    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 16, inputWidth: 16, inputDepth: 1, outputSize: 4);
        arch.RandomSeed = 42;
        return new GraFPrint<float>(arch, new GraFPrintOptions
        {
            NumMels = 16,
            GnnHiddenDim = 16,
            NumGnnLayers = 1,
            KNeighbors = 2,
            PeakFilters = 4,
            EncoderEmbeddingDim = 16,
            ProjectionExpansion = 2,
            DropoutRate = 0.0,
            LearningRate = 3e-4,
            MinimumLearningRate = 3e-6,
            LRSchedulerTMax = 100,
        });
    }
}
