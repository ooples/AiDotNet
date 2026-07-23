namespace AiDotNet.Serving.Security.Licensing;

/// <summary>
/// License key management and validation service.
/// </summary>
public interface ILicenseService
{
    Task<LicenseCreateResponse> CreateAsync(LicenseCreateRequest request, CancellationToken cancellationToken = default);

    Task<LicenseValidationResponse> ValidateAsync(LicenseValidateRequest request, CancellationToken cancellationToken = default);

    Task<bool> RevokeAsync(Guid licenseId, CancellationToken cancellationToken = default);

    Task<List<LicenseInfo>> ListAsync(CancellationToken cancellationToken = default);

    Task<LicenseInfo?> GetAsync(Guid licenseId, CancellationToken cancellationToken = default);
}
