namespace AiDotNet.Models;

public class ModelMetadata<T>
{
    public ModelType ModelType { get; set; }
    public int FeatureCount { get; set; }
    public int Complexity { get; set; }
    public string Description { get; set; } = string.Empty;
    public Dictionary<string, object> AdditionalInfo { get; set; } = [];
    public byte[] ModelData { get; set; } = [];
}