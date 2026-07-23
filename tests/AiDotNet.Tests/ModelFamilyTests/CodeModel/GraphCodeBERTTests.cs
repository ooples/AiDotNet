using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Engines;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.CodeModel;

/// <summary>
/// Manual reduced-scale factory for <see cref="GraphCodeBERT{T}"/>.
/// </summary>
/// <remarks>
/// GraphCodeBERT is excluded from the auto-generator (see
/// TestScaffoldGenerator.ExcludedClassNames): its parameterless ctor builds the
/// CodeSynthesisArchitecture defaults (BERT-base scale — 6 layers, 512 dim, 512
/// seq, 50000 vocab), whose 512→50000 output projection makes the ~250-step
/// training invariants overflow the 120s CI budget. Per Guo et al. 2021
/// ("GraphCodeBERT: Pre-training Code Representations with Data Flow",
/// arXiv:2009.08366) GraphCodeBERT is a CodeBERT extension with a data-flow
/// graph; this scaffold uses the same architecture shape as the CodeBERTTests
/// smoke config (2 encoder layers, 64 dim, 4 heads, 128 vocab) with
/// UseDataFlow=true so the graph-convolution path is exercised, training in
/// seconds on CPU.
/// </remarks>
public class GraphCodeBERTTests : NeuralNetworkModelTestBase
{
    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        var arch = new CodeSynthesisArchitecture<double>(
            synthesisType: SynthesisType.Neural,
            targetLanguage: ProgramLanguage.Python,
            codeTaskType: CodeTask.Completion,
            numEncoderLayers: 2,
            numDecoderLayers: 0,
            numHeads: 4,
            modelDimension: 64,
            feedForwardDimension: 128,
            maxSequenceLength: 32,
            vocabularySize: 128,
            useDataFlow: true);
        return new GraphCodeBERT<double>(arch);
    }

    protected override int[] InputShape => new[] { 32 };
    protected override int[] OutputShape => new[] { 128 };
}
