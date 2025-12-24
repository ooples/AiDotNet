using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.ProgramSynthesis;
using AiDotNet.Serving.Security;
using Microsoft.Extensions.Options;
using Xunit;

namespace AiDotNet.Serving.Tests;

public sealed class ServingCodeTaskRequestValidatorTimeoutTests
{
    [Fact]
    public void TryValidate_RejectsMaxWallClockMillisecondsAboveTierLimit()
    {
        var options = Options.Create(new ServingProgramSynthesisOptions
        {
            Free = new ServingProgramSynthesisLimitOptions { MaxTaskTimeSeconds = 2 },
            Premium = new ServingProgramSynthesisLimitOptions { MaxTaskTimeSeconds = 5 },
            Enterprise = new ServingProgramSynthesisLimitOptions { MaxTaskTimeSeconds = 15 }
        });

        var validator = new ServingCodeTaskRequestValidator(options);

        var request = new CodeSummarizationRequest
        {
            Language = ProgramLanguage.CSharp,
            Code = "class C {}",
            MaxWallClockMilliseconds = 2500
        };

        var ctx = new ServingRequestContext { Tier = ServingTier.Free, IsAuthenticated = false };

        Assert.False(validator.TryValidate(request, ctx, out var error));
        Assert.Contains("MaxWallClockMilliseconds exceeds tier limit", error);
    }
}

