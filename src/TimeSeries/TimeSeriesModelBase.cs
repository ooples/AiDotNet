namespace AiDotNet.TimeSeries;

public abstract class TimeSeriesModelBase<T> : ITimeSeriesModel<T>
{
    protected TimeSeriesRegressionOptions<T> _options;
    protected INumericOperations<T> NumOps;

    protected TimeSeriesModelBase(TimeSeriesRegressionOptions<T> options)
    {
        _options = options;
        NumOps = MathHelper.GetNumericOperations<T>();
    }

    public abstract void Train(Matrix<T> x, Vector<T> y);
    public abstract Vector<T> Predict(Matrix<T> input);
    public abstract Dictionary<string, T> EvaluateModel(Matrix<T> xTest, Vector<T> yTest);

    public virtual byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize common options
        writer.Write(_options.LagOrder);
        writer.Write(_options.IncludeTrend);
        writer.Write(_options.SeasonalPeriod);
        writer.Write(_options.AutocorrelationCorrection);
        writer.Write((int)_options.ModelType);

        // Let derived classes serialize their specific data
        SerializeCore(writer);

        return ms.ToArray();
    }

    public virtual void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);

        // Deserialize common options
        _options.LagOrder = reader.ReadInt32();
        _options.IncludeTrend = reader.ReadBoolean();
        _options.SeasonalPeriod = reader.ReadInt32();
        _options.AutocorrelationCorrection = reader.ReadBoolean();
        _options.ModelType = (TimeSeriesModelType)reader.ReadInt32();

        // Let derived classes deserialize their specific data
        DeserializeCore(reader);
    }

    protected abstract void SerializeCore(BinaryWriter writer);
    protected abstract void DeserializeCore(BinaryReader reader);
}