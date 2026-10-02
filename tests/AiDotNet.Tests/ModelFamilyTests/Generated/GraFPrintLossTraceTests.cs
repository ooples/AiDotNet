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
            // The paper's schedule (Adam, 8e-5 cosine-annealed to 7e-7), the GraFPrintOptions defaults. At 3e-4
            // the 10-step conformance runs were chaotic (training loss 1.40, 2.11, 0.40, 1.24): the dynamic k-NN
            // graph and batch-4 BatchNorm statistics turned ULP-level differences into opposite verdicts, so
            // the same weights passed eager and failed compiled on Linux. At 8e-5 both paths agree step for step.
            LearningRate = 8e-5,
            MinimumLearningRate = 7e-7,
            LRSchedulerTMax = 100,
        });
    }
}
