namespace AiDotNet.Models;

public class NormalizationInfo<T>
{
    public INormalizer<T>? Normalizer { get; set; }
    public List<NormalizationParameters<T>> XParams { get; set; } = [];
    public NormalizationParameters<T> YParams { get; set; } = new();
}