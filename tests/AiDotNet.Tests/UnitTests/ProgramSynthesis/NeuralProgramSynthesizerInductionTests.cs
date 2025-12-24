using System;
using System.Collections.Generic;
using System.Threading;
using AiDotNet.ProgramSynthesis.Engines;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Interfaces;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.Tests.UnitTests.ProgramSynthesis.Fakes;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ProgramSynthesis;

public sealed class NeuralProgramSynthesizerInductionTests
{
    [Fact]
    public void SynthesizeProgram_WithExamples_RefinesWhenFeedbackImprovesFitness()
    {
        var architecture = new CodeSynthesisArchitecture<double>(
            synthesisType: SynthesisType.Neural,
            targetLanguage: ProgramLanguage.Generic,
            codeTaskType: CodeTask.Generation,
            numEncoderLayers: 0,
            numDecoderLayers: 0,
            numHeads: 1,
            modelDimension: 16,
            feedForwardDimension: 32,
            maxSequenceLength: 64,
            vocabularySize: 256,
            maxProgramLength: 500,
            dropoutRate: 0.0,
            usePositionalEncoding: false);

        var codeModel = FakeCodeModel.CreateDefault(ProgramLanguage.Generic);
        var executor = new FeedbackSensitiveExecutionEngine();

        var synthesizer = new NeuralProgramSynthesizer<double>(architecture, codeModel, executionEngine: executor);

        var input = new ProgramInput<double>
        {
            TargetLanguage = ProgramLanguage.Generic,
            Description = "Return the input unchanged.",
            Examples = new List<ProgramInputOutputExample>
            {
                new() { Input = "hello", ExpectedOutput = "hello" }
            }
        };

        var result = synthesizer.SynthesizeProgram(input);

        Assert.True(result.IsValid);
        Assert.Equal(1.0, result.FitnessScore, precision: 6);
    }

    private sealed class FeedbackSensitiveExecutionEngine : IProgramExecutionEngine
    {
        private string? _initialSource;

        public bool TryExecute(
            ProgramLanguage language,
            string sourceCode,
            string input,
            out string output,
            out string? errorMessage,
            CancellationToken cancellationToken = default)
        {
            errorMessage = null;

            _initialSource ??= sourceCode;

            // The first synthesized program is treated as the "initial" candidate and fails. Any refinement that produces
            // a different source string is treated as an improvement and passes.
            output = string.Equals(sourceCode, _initialSource, StringComparison.Ordinal) ? "wrong" : input;
            return true;
        }
    }
}
