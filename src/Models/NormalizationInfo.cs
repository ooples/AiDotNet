namespace AiDotNet.Models;

public class NormalizationInfo
{
    public List<NormalizationParameters> XParams { get; set; } = [];
    public NormalizationParameters YParams { get; set; } = new();
}