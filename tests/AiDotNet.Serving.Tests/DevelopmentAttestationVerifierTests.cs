using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Security.Attestation;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using Xunit;

namespace AiDotNet.Serving.Tests;

public class DevelopmentAttestationVerifierTests
{
    [Fact]
    public void Constructor_Throws_WhenDependenciesNull()
    {
        var environment = new TestHostEnvironment();
        var options = Options.Create(new AttestationOptions());

        Assert.Throws<ArgumentNullException>(() => new DevelopmentAttestationVerifier(environment: null!, options));
        Assert.Throws<ArgumentNullException>(() => new DevelopmentAttestationVerifier(environment, options: null!));
    }

    [Fact]
    public async Task VerifyAsync_ReturnsFailure_WhenEvidenceNull()
    {
        var environment = new TestHostEnvironment { EnvironmentName = Environments.Production };
        var options = Options.Create(new AttestationOptions());
        var verifier = new DevelopmentAttestationVerifier(environment, options);

        var result = await verifier.VerifyAsync(evidence: null!);

        Assert.False(result.IsSuccess);
        Assert.False(string.IsNullOrWhiteSpace(result.FailureReason));
    }

    [Fact]
    public async Task VerifyAsync_AllowsUnverifiedEvidence_WhenDevelopmentAndEnabled()
    {
        var environment = new TestHostEnvironment { EnvironmentName = Environments.Development };
        var options = Options.Create(new AttestationOptions { AllowUnverifiedAttestationInDevelopment = true });
        var verifier = new DevelopmentAttestationVerifier(environment, options);

        var result = await verifier.VerifyAsync(new AttestationEvidence { AttestationToken = "ignored" });

        Assert.True(result.IsSuccess);
    }

    [Fact]
    public async Task VerifyAsync_AllowsStaticToken_WhenConfigured()
    {
        var environment = new TestHostEnvironment { EnvironmentName = Environments.Production };
        var options = Options.Create(new AttestationOptions { StaticTestToken = "token" });
        var verifier = new DevelopmentAttestationVerifier(environment, options);

        var result = await verifier.VerifyAsync(new AttestationEvidence { AttestationToken = "token" });

        Assert.True(result.IsSuccess);
    }

    [Fact]
    public async Task VerifyAsync_ReturnsFailure_WhenNoRuleMatches()
    {
        var environment = new TestHostEnvironment { EnvironmentName = Environments.Production };
        var options = Options.Create(new AttestationOptions());
        var verifier = new DevelopmentAttestationVerifier(environment, options);

        var result = await verifier.VerifyAsync(new AttestationEvidence { AttestationToken = "token" });

        Assert.False(result.IsSuccess);
        Assert.False(string.IsNullOrWhiteSpace(result.FailureReason));
    }
}
