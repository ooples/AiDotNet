using System;
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Agentic.Tools;

/// <summary>
/// Exposes a trained model's prediction as a callable agent tool, so an LLM agent can query the model you
/// just trained mid-reasoning — the same way it would call web search or a calculator.
/// </summary>
/// <remarks>
/// <para>
/// The tool takes a numeric feature vector as <c>{ "features": [ ... ] }</c> and returns the model's output
/// vector as JSON. It is built by <c>AiModelResult</c> for tabular models (matrix/vector shaped), which supplies
/// the marshalling delegate; unsupported model shapes simply do not produce this tool.
/// </para>
/// <para><b>For Beginners:</b> This turns your trained model into a "gadget" the assistant can use. When the
/// assistant needs a prediction, it calls this tool with the input numbers and gets the model's answer back.</para>
/// </remarks>
public sealed class ModelPredictionTool : AgentToolBase
{
    private readonly Func<double[], double[]> _predict;
    private readonly int _featureCount;

    /// <summary>Creates a model-prediction tool.</summary>
    /// <param name="predict">Maps a feature vector to the model's output vector.</param>
    /// <param name="featureCount">The number of features the model expects (for the schema and validation).</param>
    /// <param name="name">The tool name the model references. Defaults to <c>predict_model</c>.</param>
    /// <param name="description">The tool description shown to the model.</param>
    public ModelPredictionTool(
        Func<double[], double[]> predict,
        int featureCount,
        string name = "predict_model",
        string? description = null)
        : base(name,
            description ?? $"Runs the trained model on a numeric feature vector of length {featureCount} and returns its prediction.",
            BuildSchema(featureCount))
    {
        _predict = predict ?? throw new ArgumentNullException(nameof(predict));
        _featureCount = featureCount;
    }

    private static JObject BuildSchema(int featureCount) => new()
    {
        ["type"] = "object",
        ["properties"] = new JObject
        {
            ["features"] = new JObject
            {
                ["type"] = "array",
                ["items"] = new JObject { ["type"] = "number" },
                ["description"] = $"The model's input features, in order ({featureCount} values).",
            },
        },
        ["required"] = new JArray("features"),
    };

    /// <inheritdoc/>
    protected override Task<ToolInvocationResult> InvokeCoreAsync(JObject arguments, CancellationToken cancellationToken)
    {
        if (arguments["features"] is not JArray featuresJson)
        {
            return Task.FromResult(ToolInvocationResult.Error("Missing required 'features' array."));
        }

        var features = new double[featuresJson.Count];
        for (int i = 0; i < featuresJson.Count; i++)
        {
            if (featuresJson[i].Type is JTokenType.Integer or JTokenType.Float)
            {
                features[i] = featuresJson[i].Value<double>();
            }
            else
            {
                return Task.FromResult(ToolInvocationResult.Error($"Feature at index {i} is not a number."));
            }
        }

        if (_featureCount > 0 && features.Length != _featureCount)
        {
            return Task.FromResult(ToolInvocationResult.Error(
                $"Expected {_featureCount} features but received {features.Length}."));
        }

        double[] outputs;
        try { outputs = _predict(features); }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            return Task.FromResult(ToolInvocationResult.Error($"Model prediction failed: {ex.Message}"));
        }

        var outArray = new JArray(outputs.Select(v => (object)v));
        var payload = new JObject { ["prediction"] = outArray };
        return Task.FromResult(ToolInvocationResult.Success(payload.ToString(Newtonsoft.Json.Formatting.None)));
    }

    /// <summary>Parses the first numeric value from this tool's JSON result, for grounded verification.</summary>
    /// <param name="toolResultContent">The tool's returned content.</param>
    /// <returns>The first prediction value, or <c>null</c> if it could not be parsed.</returns>
    public static double? TryParseFirstPrediction(string toolResultContent)
    {
        if (string.IsNullOrWhiteSpace(toolResultContent)) return null;
        try
        {
            var obj = JObject.Parse(toolResultContent);
            if (obj["prediction"] is JArray arr && arr.Count > 0 &&
                arr[0].Type is JTokenType.Integer or JTokenType.Float)
            {
                return arr[0].Value<double>();
            }
        }
        catch (Exception ex) when (ex is Newtonsoft.Json.JsonException or FormatException)
        {
        }

        return null;
    }
}
