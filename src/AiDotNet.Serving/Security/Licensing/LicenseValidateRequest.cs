using System.ComponentModel.DataAnnotations;

namespace AiDotNet.Serving.Security.Licensing;

/// <summary>
/// Request to validate a license key. Sent by the client-side LicenseValidator.
/// </summary>
public sealed class LicenseValidateRequest
{
    [Required(AllowEmptyStrings = false)]
    [StringLength(512)]
    public string Key { get; set; } = string.Empty;

    [StringLength(256)]
    public string? MachineId { get; set; }

    [StringLength(256)]
    public string? MachineName { get; set; }

    [StringLength(64)]
    public string? Environment { get; set; }
}
