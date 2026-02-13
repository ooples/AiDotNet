using System.IdentityModel.Tokens.Jwt;
using System.Security.Cryptography.X509Certificates;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using Microsoft.IdentityModel.Tokens;
using AiDotNet.Validation;

namespace AiDotNet.Serving.Security.Attestation;

/// <summary>
/// Development-friendly attestation verifier.
/// </summary>
public sealed class DevelopmentAttestationVerifier : IAttestationVerifier
{
    private readonly IHostEnvironment _environment;
    private readonly AttestationOptions _options;
    private readonly JwtSecurityTokenHandler _tokenHandler = new();
    private readonly TokenValidationParameters? _jwtParameters;
    private readonly JwtAttestationOptions? _jwtOptions;

    public DevelopmentAttestationVerifier(IHostEnvironment environment, IOptions<AttestationOptions> options)
    {
        Guard.NotNull(environment);
        _environment = environment;
        _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
        _jwtOptions = _options.Jwt;
        _jwtParameters = _jwtOptions != null ? BuildJwtValidationParameters(_jwtOptions) : null;
    }

    public Task<AttestationVerificationResult> VerifyAsync(AttestationEvidence evidence, CancellationToken cancellationToken = default)
    {
        if (evidence == null)
        {
            return Task.FromResult(new AttestationVerificationResult(false, "Attestation evidence is required."));
        }

        var allowListResult = ValidateEvidenceAllowLists(evidence);
        if (!allowListResult.IsSuccess)
        {
            return Task.FromResult(allowListResult);
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

        if (_jwtOptions != null && _jwtParameters != null)
        {
            return Task.FromResult(VerifyJwtAttestation(evidence));
        }

        return Task.FromResult(new AttestationVerificationResult(false, "No attestation verifier is configured for this environment."));
    }

    private AttestationVerificationResult ValidateEvidenceAllowLists(AttestationEvidence evidence)
    {
        if (_options.AllowedPlatforms.Length > 0)
        {
            if (string.IsNullOrWhiteSpace(evidence.Platform))
            {
                return new AttestationVerificationResult(false, "Attestation platform is required.");
            }

            if (!ContainsIgnoreCase(_options.AllowedPlatforms, evidence.Platform))
            {
                return new AttestationVerificationResult(false, $"Platform '{evidence.Platform}' is not allowed.");
            }
        }

        if (_options.AllowedTeeTypes.Length > 0)
        {
            if (string.IsNullOrWhiteSpace(evidence.TeeType))
            {
                return new AttestationVerificationResult(false, "Attestation teeType is required.");
            }

            if (!ContainsIgnoreCase(_options.AllowedTeeTypes, evidence.TeeType))
            {
                return new AttestationVerificationResult(false, $"TEE type '{evidence.TeeType}' is not allowed.");
            }
        }

        return new AttestationVerificationResult(true);
    }

    private AttestationVerificationResult VerifyJwtAttestation(AttestationEvidence evidence)
    {
        if (string.IsNullOrWhiteSpace(evidence.AttestationToken))
        {
            return new AttestationVerificationResult(false, "Attestation token is required.");
        }

        try
        {
            var principal = _tokenHandler.ValidateToken(evidence.AttestationToken, _jwtParameters!, out _);

            if (_jwtOptions!.RequireNonceClaim)
            {
                if (string.IsNullOrWhiteSpace(evidence.Nonce))
                {
                    return new AttestationVerificationResult(false, "Attestation nonce is required.");
                }

                var nonce = principal.FindFirst(_jwtOptions.NonceClaimType)?.Value;
                if (string.IsNullOrWhiteSpace(nonce) || !string.Equals(nonce, evidence.Nonce, StringComparison.Ordinal))
                {
                    return new AttestationVerificationResult(false, "Attestation nonce claim did not match.");
                }
            }

            if (_jwtOptions.RequirePlatformClaimMatch)
            {
                if (string.IsNullOrWhiteSpace(evidence.Platform))
                {
                    return new AttestationVerificationResult(false, "Attestation platform is required.");
                }

                var platform = principal.FindFirst(_jwtOptions.PlatformClaimType)?.Value;
                if (string.IsNullOrWhiteSpace(platform) || !string.Equals(platform, evidence.Platform, StringComparison.OrdinalIgnoreCase))
                {
                    return new AttestationVerificationResult(false, "Attestation platform claim did not match.");
                }
            }

            if (_jwtOptions.RequireTeeTypeClaimMatch)
            {
                if (string.IsNullOrWhiteSpace(evidence.TeeType))
                {
                    return new AttestationVerificationResult(false, "Attestation teeType is required.");
                }

                var teeType = principal.FindFirst(_jwtOptions.TeeTypeClaimType)?.Value;
                if (string.IsNullOrWhiteSpace(teeType) || !string.Equals(teeType, evidence.TeeType, StringComparison.OrdinalIgnoreCase))
                {
                    return new AttestationVerificationResult(false, "Attestation teeType claim did not match.");
                }
            }

            return new AttestationVerificationResult(true);
        }
        catch (Exception ex) when (
            ex is SecurityTokenException ||
            ex is ArgumentException ||
            ex is InvalidOperationException)
        {
            return new AttestationVerificationResult(false, "Attestation token validation failed.");
        }
    }

    private static TokenValidationParameters BuildJwtValidationParameters(JwtAttestationOptions jwtOptions)
    {
        var signingKeys = ResolveSigningKeys(jwtOptions);

        if (signingKeys.Length == 0)
        {
            throw new InvalidOperationException("Jwt attestation is configured but no signing keys were provided.");
        }

        return new TokenValidationParameters
        {
            ValidateIssuer = jwtOptions.ValidIssuers.Length > 0,
            ValidIssuers = jwtOptions.ValidIssuers,
            ValidateAudience = jwtOptions.ValidAudiences.Length > 0,
            ValidAudiences = jwtOptions.ValidAudiences,
            ValidateIssuerSigningKey = true,
            IssuerSigningKeys = signingKeys,
            ValidateLifetime = true,
            ClockSkew = TimeSpan.FromSeconds(Math.Max(0, jwtOptions.ClockSkewSeconds))
        };
    }

    private static SecurityKey[] ResolveSigningKeys(JwtAttestationOptions jwtOptions)
    {
        var keys = new List<SecurityKey>();

        foreach (var base64 in jwtOptions.TrustedSigningCertificatesBase64 ?? [])
        {
            if (string.IsNullOrWhiteSpace(base64))
            {
                continue;
            }

            var bytes = Convert.FromBase64String(base64);
            var cert = new X509Certificate2(bytes);
            keys.Add(new X509SecurityKey(cert));
        }

        foreach (var path in jwtOptions.TrustedSigningCertificatePaths ?? [])
        {
            if (string.IsNullOrWhiteSpace(path))
            {
                continue;
            }

            var cert = new X509Certificate2(path);
            keys.Add(new X509SecurityKey(cert));
        }

        return keys.ToArray();
    }

    private static bool ContainsIgnoreCase(string[] allowed, string value)
    {
        foreach (var candidate in allowed)
        {
            if (string.Equals(candidate?.Trim(), value?.Trim(), StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }
}
