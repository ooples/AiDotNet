using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Tools;
using Newtonsoft.Json.Linq;

using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Agents;

/// <summary>
/// Wraps an agent so the trained model grounds its answers: after the inner agent runs, any prediction it
/// obtained from the model-prediction tool is recomputed authoritatively and checked against the numbers in the
/// agent's final answer. If the answer contradicts the model, the agent is told the model's actual prediction and
/// asked to revise — up to a bounded number of times.
/// </summary>
/// <typeparam name="T">The numeric type shared with the underlying chat client.</typeparam>
/// <remarks>
/// <para>
/// Only a combined ML + agent library can do this: the LLM cannot talk its way past the model it just queried.
/// The check is deliberately conservative — it fires only when the agent actually used the model tool and its
/// stated number disagrees with the model's real output, so it corrects hallucinated numbers without second-
/// guessing answers the model has nothing to say about.
/// </para>
/// <para><b>For Beginners:</b> This makes the assistant honest about the model's output. If the assistant asks
/// the model for a prediction and then reports a different number, this catches it and makes the assistant fix it.</para>
/// </remarks>
public sealed class GroundedVerifierAgent<T> : IAgent<T>
{
    private static readonly Regex NumberPattern = new(@"-?\d+(?:\.\d+)?", RegexOptions.Compiled);

    private readonly IAgent<T> _inner;
    private readonly Func<double[], double[]> _predict;
    private readonly string _modelToolName;
    private readonly int _maxRefinements;
    private readonly double _relativeTolerance;

    /// <summary>Wraps an agent with model-grounded verification.</summary>
    /// <param name="inner">The agent whose answers should be grounded.</param>
    /// <param name="predict">The authoritative model prediction (same model the tool wraps).</param>
    /// <param name="modelToolName">The model-prediction tool's name, used to find the agent's model calls.</param>
    /// <param name="maxRefinements">How many times the agent may be asked to revise. Non-positive means one.</param>
    /// <param name="relativeTolerance">Allowed relative gap between the agent's number and the model's.</param>
    public GroundedVerifierAgent(
        IAgent<T> inner,
        Func<double[], double[]> predict,
        string modelToolName,
        int maxRefinements = 1,
        double relativeTolerance = 0.01)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
        _predict = predict ?? throw new ArgumentNullException(nameof(predict));
        _modelToolName = string.IsNullOrWhiteSpace(modelToolName) ? "predict_model" : modelToolName;
        _maxRefinements = maxRefinements > 0 ? maxRefinements : 1;
        _relativeTolerance = relativeTolerance > 0 ? relativeTolerance : 0.01;
    }

    /// <inheritdoc/>
    public string Name => _inner.Name;

    /// <inheritdoc/>
    public string Description => _inner.Description;

    /// <inheritdoc/>
    public async Task<AgentRunResult> RunAsync(
        IReadOnlyList<ChatMessage> messages, CancellationToken cancellationToken = default)
    {
        var result = await _inner.RunAsync(messages, cancellationToken).ConfigureAwait(false);

        for (int attempt = 0; attempt < _maxRefinements; attempt++)
        {
            var modelValues = ExtractModelPredictions(result.Messages);
            if (modelValues.Count == 0)
            {
                return result; // the agent never consulted the model; nothing to ground against.
            }

            var stated = ExtractNumbers(result.FinalText);
            bool grounded = modelValues.Any(mv => stated.Any(s => IsClose(s, mv)));
            if (grounded)
            {
                return result;
            }

            // The answer contradicts the model. Tell the agent the model's real prediction and let it revise.
            var correction = ChatMessage.User(
                "Your answer is inconsistent with the model's own prediction. The trained model actually predicted: " +
                string.Join(", ", modelValues.Select(v => v.ToString("G6", CultureInfo.InvariantCulture))) +
                ". Revise your final answer to be consistent with the model's prediction.");

            var next = new List<ChatMessage>(result.Messages) { correction };
            result = await _inner.RunAsync(next, cancellationToken).ConfigureAwait(false);
        }

        return result;
    }

    private List<double> ExtractModelPredictions(IReadOnlyList<ChatMessage> transcript)
    {
        var values = new List<double>();
        foreach (var message in transcript)
        {
            if (message.Role != ChatRole.Assistant) continue;
            foreach (var content in message.Contents)
            {
                if (content is not ToolCallContent call || !string.Equals(call.ToolName, _modelToolName, StringComparison.Ordinal))
                {
                    continue;
                }

                var features = ParseFeatures(call.ArgumentsJson);
                if (features is null) continue;
                try
                {
                    var outputs = _predict(features);
                    if (outputs.Length > 0) values.Add(outputs[0]);
                }
                catch (Exception ex) when (ex is not OperationCanceledException)
                {
                }
            }
        }

        return values;
    }

    private static double[]? ParseFeatures(string argumentsJson)
    {
        try
        {
            var obj = Newtonsoft.Json.Linq.JObject.Parse(argumentsJson);
            if (obj["features"] is not Newtonsoft.Json.Linq.JArray arr) return null;
            var features = new double[arr.Count];
            for (int i = 0; i < arr.Count; i++)
            {
                if (arr[i].Type is not (Newtonsoft.Json.Linq.JTokenType.Integer or Newtonsoft.Json.Linq.JTokenType.Float))
                {
                    return null;
                }

                features[i] = arr[i].Value<double>();
            }

            return features;
        }
        catch (Exception ex) when (ex is Newtonsoft.Json.JsonException or FormatException)
        {
            return null;
        }
    }

    private static List<double> ExtractNumbers(string text)
    {
        var numbers = new List<double>();
        if (string.IsNullOrEmpty(text)) return numbers;
        foreach (Match m in NumberPattern.Matches(text))
        {
            if (double.TryParse(m.Value, NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
            {
                numbers.Add(v);
            }
        }

        return numbers;
    }

    private bool IsClose(double a, double b) => Math.Abs(a - b) <= _relativeTolerance * Math.Max(1.0, Math.Abs(b));
}
