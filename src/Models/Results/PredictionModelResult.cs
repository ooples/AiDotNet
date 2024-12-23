global using Newtonsoft.Json;
global using Formatting = Newtonsoft.Json.Formatting;

namespace AiDotNet.Models.Results;

[Serializable]
internal class PredictionModelResult<T> : IPredictiveModel<T>
{
    public IRegression<T>? Model { get; private set; }
    public OptimizationResult<T> OptimizationResult { get; private set; } = new();
    public NormalizationInfo<T> NormalizationInfo { get; private set; } = new();
    public ModelMetadata<T> ModelMetadata { get; private set; } = new();

    public PredictionModelResult(IRegression<T> model, OptimizationResult<T> optimizationResult, NormalizationInfo<T> normalizationInfo)
    {
        Model = model;
        OptimizationResult = optimizationResult;
        NormalizationInfo = normalizationInfo;
        ModelMetadata = model.GetModelMetadata();
    }

    public PredictionModelResult()
    {
    }

    public ModelMetadata<T> GetModelMetadata()
    {
        return ModelMetadata;
    }

    public Vector<T> Predict(Matrix<T> newData)
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Model is not initialized.");
        }

        if (NormalizationInfo.Normalizer == null)
        {
            throw new InvalidOperationException("Normalizer is not initialized.");
        }

        var (normalizedNewData, _) = NormalizationInfo.Normalizer.NormalizeMatrix(newData);
        var normalizedPredictions = Model.Predict(normalizedNewData);
        return NormalizationInfo.Normalizer.DenormalizeVector(normalizedPredictions, NormalizationInfo.YParams);
    }

    public byte[] Serialize()
    {
        var jsonString = JsonConvert.SerializeObject(this, Formatting.Indented, new JsonSerializerSettings
        {
            TypeNameHandling = TypeNameHandling.All
        });

        return Encoding.UTF8.GetBytes(jsonString);
    }

    public void Deserialize(byte[] data)
    {
        var jsonString = Encoding.UTF8.GetString(data);
        var deserializedObject = JsonConvert.DeserializeObject<PredictionModelResult<T>>(jsonString, new JsonSerializerSettings
        {
            TypeNameHandling = TypeNameHandling.All
        });

        if (deserializedObject != null)
        {
            Model = deserializedObject.Model;
            OptimizationResult = deserializedObject.OptimizationResult;
            NormalizationInfo = deserializedObject.NormalizationInfo;
            ModelMetadata = deserializedObject.ModelMetadata;
        }
        else
        {
            throw new InvalidOperationException("Failed to deserialize the model.");
        }
    }

    public void SaveModel(string filePath)
    {
        File.WriteAllBytes(filePath, Serialize());
    }

    public static PredictionModelResult<T> LoadModel(string filePath)
    {
        var data = File.ReadAllBytes(filePath);
        var result = new PredictionModelResult<T>();
        result.Deserialize(data);
        return result;
    }
}