using System.ComponentModel.DataAnnotations;

namespace AiDotNet.Serving.Security.Licensing;

/// <summary>
/// Request to validate a license key. Sent by the client-side LicenseValidator.
/// </summary>
public sealed class LicenseValidateRequest
{
    [Required(AllowEmptyStrings = false)]
    public string Key { get; set; } = string.Empty;

    public string? MachineId { get; set; }

    public string? MachineName { get; set; }

    [StringLength(64)]
    public string? Environment { get; set; }
}
