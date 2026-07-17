using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Tools;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Models.Results;

public partial class AiModelResult<T, TInput, TOutput>
{
    private IReadOnlyList<IAgentTool> _configuredTools = Array.Empty<IAgentTool>();

    /// <summary>The tools configured via <c>ConfigureTool(...)</c>, available to agents built from this result.</summary>
    public IReadOnlyList<IAgentTool> ConfiguredTools => _configuredTools;

    /// <summary>Records the configured agent tools. Called by the builder during Build.</summary>
    internal void SetConfiguredTools(IReadOnlyList<IAgentTool>? tools)
        => _configuredTools = tools is null ? Array.Empty<IAgentTool>() : tools.ToArray();

    /// <summary>
    /// Builds the trained model itself as a callable agent tool, or <c>null</c> when the model shape is not
    /// supported (only matrix-input tabular models are exposed as tools).
    /// </summary>
    /// <param name="name">The tool name the model references.</param>
    public ModelPredictionTool? CreateModelPredictionTool(string name = "predict_model")
    {
        var predict = TryBuildModelPredict();
        return predict is null ? null : new ModelPredictionTool(predict, featureCount: 0, name: name);
    }

    /// <summary>
    /// Creates an agent, driven by the supplied chat client, that can call every configured tool plus the trained
    /// model itself (exposed as a prediction tool). By default tool calls are schema-guarded and the agent's
    /// numeric answers are grounded against the model.
    /// </summary>
    /// <param name="chatClient">The chat client the agent drives.</param>
    /// <param name="options">Executor options (name, system prompt, iteration budget, sampling).</param>
    /// <param name="guardTools">When <c>true</c>, every tool is schema-validated before it runs.</param>
    /// <param name="groundWithModel">When <c>true</c> and the model is tool-exposable, the agent's answers are
    /// verified against the model's real predictions and revised on contradiction.</param>
    /// <returns>The agent, ready to run.</returns>
    public IAgent<T> CreateAgent(
        IChatClient<T> chatClient,
        AgentExecutorOptions? options = null,
        bool guardTools = true,
        bool groundWithModel = true)
    {
        Guard.NotNull(chatClient);

        var modelTool = CreateModelPredictionTool();
        var tools = BuildToolCollection(modelTool, guardTools);

        IAgent<T> agent = new AgentExecutor<T>(chatClient, tools, options ?? new AgentExecutorOptions());

        if (groundWithModel && modelTool is not null)
        {
            var predict = TryBuildModelPredict();
            if (predict is not null)
            {
                agent = new GroundedVerifierAgent<T>(agent, predict, modelTool.Name);
            }
        }

        return agent;
    }

    /// <summary>
    /// Runs an agent on a single task. When a <paramref name="router"/> is supplied, the request is nudged
    /// toward tools that helped on similar past requests, and the run's tool outcomes are recorded so the router
    /// keeps learning.
    /// </summary>
    /// <param name="chatClient">The chat client the agent drives.</param>
    /// <param name="task">The user's request.</param>
    /// <param name="options">Executor options.</param>
    /// <param name="guardTools">Schema-guard tool calls.</param>
    /// <param name="groundWithModel">Ground the agent's answers against the model.</param>
    /// <param name="router">Optional learned tool router for adaptive tool preference.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task<AgentRunResult> RunAgentAsync(
        IChatClient<T> chatClient,
        string task,
        AgentExecutorOptions? options = null,
        bool guardTools = true,
        bool groundWithModel = true,
        LearnedToolRouter? router = null,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(chatClient);
        Guard.NotNull(task);

        var effectiveOptions = options ?? new AgentExecutorOptions();
        if (router is not null)
        {
            var toolNames = ToolNames();
            var hint = router.BuildHint(task, toolNames);
            if (hint is not null)
            {
                effectiveOptions = WithSystemPromptSuffix(effectiveOptions, hint);
            }
        }

        var agent = CreateAgent(chatClient, effectiveOptions, guardTools, groundWithModel);
        var result = await agent.RunAsync(new[] { ChatMessage.User(task) }, cancellationToken).ConfigureAwait(false);

        if (router is not null)
        {
            // Reward tools that contributed to a completed answer; partial credit when the run stalled.
            double reward = result.Completed ? 1.0 : 0.3;
            var used = result.Messages
                .SelectMany(m => m.Contents)
                .OfType<ToolCallContent>()
                .Select(c => c.ToolName)
                .Distinct();
            foreach (var toolName in used)
            {
                router.Record(task, toolName, reward);
            }
        }

        return result;
    }

    /// <summary>
    /// Creates a leaf agent whose only tool is the trained model — a "model analyst" specialist to place inside a
    /// supervisor/swarm alongside other specialists (each of which the coordinator can call as a tool).
    /// </summary>
    /// <param name="chatClient">The chat client the analyst drives.</param>
    /// <param name="options">Executor options; a sensible name/description is supplied when omitted.</param>
    /// <returns>The analyst agent, or <c>null</c> when the model is not tool-exposable.</returns>
    public IAgent<T>? CreateModelAnalystAgent(IChatClient<T> chatClient, AgentExecutorOptions? options = null)
    {
        Guard.NotNull(chatClient);
        var modelTool = CreateModelPredictionTool();
        if (modelTool is null) return null;

        var tools = new ToolCollection();
        tools.Add(new GuardedAgentTool(modelTool));

        var opts = options ?? new AgentExecutorOptions
        {
            Name = "model_analyst",
            Description = "Answers questions by querying the trained model for predictions.",
        };
        return new AgentExecutor<T>(chatClient, tools, opts);
    }

    /// <summary>
    /// Creates a supervisor that coordinates a team of specialist agents (each exposed to the coordinator as a
    /// callable tool). Combine with <see cref="CreateModelAnalystAgent"/> to put the trained model on the team.
    /// </summary>
    /// <param name="coordinator">The chat client that drives the supervisor's routing.</param>
    /// <param name="workers">The specialist agents to coordinate.</param>
    /// <param name="options">Supervisor options.</param>
    public SupervisorAgent<T> CreateSupervisor(
        IChatClient<T> coordinator,
        IReadOnlyList<IAgent<T>> workers,
        SupervisorOptions? options = null)
    {
        Guard.NotNull(coordinator);
        Guard.NotNull(workers);
        return new SupervisorAgent<T>(coordinator, workers, options);
    }

    private ToolCollection BuildToolCollection(IAgentTool? modelTool, bool guardTools)
    {
        var tools = new ToolCollection();
        foreach (var tool in _configuredTools)
        {
            tools.Add(guardTools ? new GuardedAgentTool(tool) : tool);
        }

        if (modelTool is not null)
        {
            tools.Add(guardTools ? new GuardedAgentTool(modelTool) : modelTool);
        }

        return tools;
    }

    private IReadOnlyList<string> ToolNames()
    {
        var names = _configuredTools.Select(t => t.Name).ToList();
        var modelTool = CreateModelPredictionTool();
        if (modelTool is not null) names.Add(modelTool.Name);
        return names;
    }

    private Func<double[], double[]>? TryBuildModelPredict()
    {
        if (Model is null || typeof(TInput) != typeof(AiDotNet.Tensors.LinearAlgebra.Matrix<T>))
        {
            return null; // only matrix-input tabular models are exposed as a prediction tool.
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        return features =>
        {
            var m = new AiDotNet.Tensors.LinearAlgebra.Matrix<T>(1, features.Length);
            for (int j = 0; j < features.Length; j++) m[0, j] = numOps.FromDouble(features[j]);
            var output = Predict((TInput)(object)m);
            var vector = ConversionsHelper.ConvertToVector<T, TOutput>(output);
            var result = new double[vector.Length];
            for (int i = 0; i < vector.Length; i++) result[i] = numOps.ToDouble(vector[i]);
            return result;
        };
    }

    private static AgentExecutorOptions WithSystemPromptSuffix(AgentExecutorOptions options, string suffix)
    {
        var basePrompt = string.IsNullOrWhiteSpace(options.SystemPrompt) ? string.Empty : options.SystemPrompt + "\n\n";
        return new AgentExecutorOptions
        {
            Name = options.Name,
            Description = options.Description,
            SystemPrompt = basePrompt + suffix,
            MaxIterations = options.MaxIterations,
            Temperature = options.Temperature,
            MaxOutputTokens = options.MaxOutputTokens,
            ToolChoice = options.ToolChoice,
        };
    }
}
