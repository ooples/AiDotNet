using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;

namespace AiDotNet.Serving.Security.Attestation;

/// <summary>
/// Development-friendly attestation verifier.
/// </summary>
public sealed class DevelopmentAttestationVerifier : IAttestationVerifier
{
    private readonly IHostEnvironment _environment;
    private readonly AttestationOptions _options;

    public DevelopmentAttestationVerifier(IHostEnvironment environment, IOptions<AttestationOptions> options)
    {
        _environment = environment ?? throw new ArgumentNullException(nameof(environment));
        _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
    }

    public Task<AttestationVerificationResult> VerifyAsync(AttestationEvidence evidence, CancellationToken cancellationToken = default)
    {
        if (evidence == null)
        {
            return Task.FromResult(new AttestationVerificationResult(false, "Attestation evidence is required."));
        }

        if (_environment.IsDevelopment() && _options.AllowUnverifiedAttestationInDevelopment)
        {
            return Task.FromResult(new AttestationVerificationResult(true));
        }

        if (!string.IsNullOrWhiteSpace(_options.StaticTestToken) &&
            string.Equals(evidence.AttestationToken, _options.StaticTestToken, StringComparison.Ordinal))
        {
            return Task.FromResult(new AttestationVerificationResult(true));
        }

        return Task.FromResult(new AttestationVerificationResult(false, "Attestation verification failed."));
    }
}

