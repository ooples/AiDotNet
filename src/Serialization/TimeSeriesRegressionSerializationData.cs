namespace AiDotNet.Serialization;

[Serializable]
public class TimeSeriesRegressionSerializationData<T>
{
    public byte[] BaseData { get; set; } = [];
    public TimeSeriesRegressionOptions<T> Options { get; set; } = new();
    public string TimeSeriesModelType { get; set; } = string.Empty;
    public byte[] TimeSeriesModelData { get; set; } = [];
}
