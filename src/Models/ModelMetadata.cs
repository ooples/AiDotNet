namespace AiDotNet.Models;

public class ModelMetadata<T>
{
    public ModelType ModelType { get; set; }
    public Vector<T>? Coefficients { get; set; }
    public T? Intercept { get; set; }
    public Vector<T>? FeatureImportances { get; set; }
    public Dictionary<string, object> AdditionalInfo { get; set; } = [];
}