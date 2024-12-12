global using Newtonsoft.Json;
global using Formatting = Newtonsoft.Json.Formatting;

namespace AiDotNet.Models;

[Serializable]
public class PredictionModelResult
{
    public IRegression? Model { get; set; }
    public OptimizationResult OptimizationResult { get; set; } = new();
    public NormalizationInfo NormalizationInfo { get; set; } = new();

    public Vector<double> Predict(Matrix<double> newData)
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

    public static PredictionModelResult LoadModel(string filePath)
    {
        string jsonString = File.ReadAllText(filePath);
        return JsonConvert.DeserializeObject<PredictionModelResult>(jsonString, new JsonSerializerSettings
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

    public static PredictionModelResult DeserializeFromJson(string jsonString)
    {
        return JsonConvert.DeserializeObject<PredictionModelResult>(jsonString, new JsonSerializerSettings
        {
            TypeNameHandling = TypeNameHandling.All
        }) ?? new();
    }
}