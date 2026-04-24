using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Engines;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.CodeModel;

/// <summary>
/// Manual factory for <see cref="CodeBERT{T}"/>. The auto-generator emits a
/// <c>NotImplementedException</c> placeholder for any model whose first
/// constructor parameter is a NeuralNetworkArchitecture <em>subclass</em>
/// (CodeSynthesisArchitecture in this case) — see TestScaffoldGenerator.
/// </summary>
/// <remarks>
/// Per Feng et al. 2020 ("CodeBERT: A Pre-Trained Model for Programming and
/// Natural Languages"), CodeBERT is a 12-layer encoder-only transformer with
/// 768 hidden, 12 heads. The test config below uses a much smaller smoke
/// shape (encoder layers=2, model dim=64, heads=4, vocab=128) so the test
/// compiles and trains inside the 60s smoke-suite budget; the full paper
/// scale is exercised by integration tests, not the auto-generated scaffold.
/// </remarks>
public class CodeBERTTests : NeuralNetworkModelTestBase
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
            vocabularySize: 128);
        return new CodeBERT<double>(arch);
    }

    protected override int[] InputShape => new[] { 32 };
    protected override int[] OutputShape => new[] { 128 };
}
