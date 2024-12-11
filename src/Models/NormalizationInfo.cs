namespace AiDotNet.Models;

public class NormalizationInfo
{
    public INormalizer? Normalizer { get; set; }
    public List<NormalizationParameters> XParams { get; set; } = [];
    public NormalizationParameters YParams { get; set; } = new();
}