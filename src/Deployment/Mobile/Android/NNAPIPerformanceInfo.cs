namespace AiDotNet.Deployment.Mobile.Android;

/// <summary>
/// Performance information for NNAPI.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> NNAPIPerformanceInfo provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class NNAPIPerformanceInfo
{
    public List<string> SupportedOperations { get; set; } = new();
    public string PreferredDevice { get; set; } = string.Empty;
    public bool SupportsInt8 { get; set; }
    public bool SupportsFp16 { get; set; }
    public bool SupportsRelaxedFp32 { get; set; }
}
