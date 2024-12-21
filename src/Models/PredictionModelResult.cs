global using Newtonsoft.Json;
global using Formatting = Newtonsoft.Json.Formatting;

namespace AiDotNet.Models;

[Serializable]
public class PredictionModelResult<T> : IPredictiveModel<T>
{
    public IRegression<T>? Model { get; set; }
    public OptimizationResult<T> OptimizationResult { get; set; } = new();
    public NormalizationInfo<T> NormalizationInfo { get; set; } = new();

    public PredictionModelResult(IRegression<T> model, OptimizationResult<T> optimizationResult, NormalizationInfo<T> normalizationInfo)
    {
        Model = model;
        OptimizationResult = optimizationResult;
        NormalizationInfo = normalizationInfo;
    }

    public PredictionModelResult()
    {
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

        // Preprocess the new data using the same normalization
        var (normalizedNewData, _) = NormalizationInfo.Normalizer.NormalizeMatrix(newData);
    
        // Make predictions
        var normalizedPredictions = Model.Predict(normalizedNewData);
    
        // Denormalize the predictions
        return NormalizationInfo.Normalizer.DenormalizeVector(normalizedPredictions, NormalizationInfo.YParams);
    }

    public void SaveModel(string filePath)
    {
        string jsonString = JsonConvert.SerializeObject(this, Formatting.Indented, new JsonSerializerSettings
        {
            TypeNameHandling = TypeNameHandling.All
        });
        File.WriteAllText(filePath, jsonString);
    }

    public static PredictionModelResult<T> LoadModel(string filePath)
    {
        string jsonString = File.ReadAllText(filePath);
        return JsonConvert.DeserializeObject<PredictionModelResult<T>>(jsonString, new JsonSerializerSettings
        {
            TypeNameHandling = TypeNameHandling.All
        }) ?? new();
    }

    public string SerializeToJson()
    {
        return JsonConvert.SerializeObject(this, Formatting.Indented, new JsonSerializerSettings
        {
            TypeNameHandling = TypeNameHandling.All
        });
    }

    public static PredictionModelResult<T> DeserializeFromJson(string jsonString)
    {
        return JsonConvert.DeserializeObject<PredictionModelResult<T>>(jsonString, new JsonSerializerSettings
        {
            TypeNameHandling = TypeNameHandling.All
        }) ?? new();
    }
}