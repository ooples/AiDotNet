using System.Reflection;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Agentic.Tools;

/// <summary>
/// An <see cref="IAgentTool"/> backed by an ordinary C# method (a delegate or a reflected method on an
/// instance). The parameter schema is generated automatically, arguments are bound from the model's JSON,
/// and the return value is serialized back to text.
/// </summary>
/// <remarks>
/// <para>
/// This is the bridge from "a method you already have" to "a tool the model can call". Supported return
/// shapes: <c>void</c>, a value, <see cref="Task"/>, or <see cref="Task{TResult}"/>. A
/// <see cref="CancellationToken"/> parameter (if present) is supplied by the runtime and hidden from the
/// model's schema.
/// </para>
/// <para><b>For Beginners:</b> Hand this a method like <c>int Add(int a, int b)</c> and it becomes a tool:
/// the model sees inputs <c>a</c> and <c>b</c>, and when it calls the tool with <c>{"a":2,"b":3}</c> this
/// class converts that JSON into real arguments, runs your method, and turns the result back into text.
/// </para>
/// </remarks>
public sealed class DelegateAgentTool : AgentToolBase
{
    private readonly MethodInfo _method;
    private readonly object? _target;

    /// <summary>
    /// Initializes a tool from a method and optional target instance.
    /// </summary>
    /// <param name="name">The tool name exposed to the model.</param>
    /// <param name="description">A description of what the tool does.</param>
    /// <param name="method">The method to invoke.</param>
    /// <param name="target">The instance to invoke the method on, or <c>null</c> for a static method.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="method"/> is <c>null</c>.</exception>
    public DelegateAgentTool(string name, string description, MethodInfo method, object? target = null)
        : base(name, description, BuildSchema(method))
    {
        _method = method;
        _target = target;
    }

    /// <summary>
    /// Initializes a tool from a delegate.
    /// </summary>
    /// <param name="name">The tool name exposed to the model.</param>
    /// <param name="description">A description of what the tool does.</param>
    /// <param name="function">The delegate to invoke.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="function"/> is <c>null</c>.</exception>
    public DelegateAgentTool(string name, string description, Delegate function)
        : this(name, description, GetMethod(function), GetTarget(function))
    {
    }

    /// <inheritdoc/>
    protected override async Task<ToolInvocationResult> InvokeCoreAsync(JObject arguments, CancellationToken cancellationToken)
    {
        var parameters = _method.GetParameters();
        var callArgs = new object?[parameters.Length];

        for (int i = 0; i < parameters.Length; i++)
        {
            var parameter = parameters[i];

            if (parameter.ParameterType == typeof(CancellationToken))
            {
                callArgs[i] = cancellationToken;
                continue;
            }

            var name = parameter.Name ?? "arg";

            if (arguments.TryGetValue(name, out var token) && token.Type != JTokenType.Null)
            {
                try
                {
                    callArgs[i] = token.ToObject(parameter.ParameterType);
                }
                catch (Exception ex) when (ex is not OutOfMemoryException && ex is not StackOverflowException)
                {
                    // Covers JsonException, FormatException, OverflowException, InvalidCastException, etc.
                    return ToolInvocationResult.Error($"Argument '{name}' could not be converted to {parameter.ParameterType.Name}: {ex.Message}");
                }
            }
            else if (parameter.HasDefaultValue)
            {
                callArgs[i] = parameter.DefaultValue;
            }
            else if (!parameter.ParameterType.IsValueType || Nullable.GetUnderlyingType(parameter.ParameterType) is not null)
            {
                callArgs[i] = null;
            }
            else
            {
                return ToolInvocationResult.Error($"Missing required parameter '{name}'.");
            }
        }

        object? returnValue;
        try
        {
            returnValue = _method.Invoke(_target, callArgs);
        }
        catch (TargetInvocationException ex) when (ex.InnerException is not null)
        {
            if (ex.InnerException is OperationCanceledException) throw ex.InnerException;
            return ToolInvocationResult.Error($"Tool '{Name}' failed: {ex.InnerException.Message}");
        }

        var result = await UnwrapResultAsync(returnValue).ConfigureAwait(false);
        var content = result switch
        {
            null => string.Empty,
            string s => s,
            _ => JsonConvert.SerializeObject(result)
        };

        return ToolInvocationResult.Success(content);
    }

    private static JObject BuildSchema(MethodInfo method)
    {
        Guard.NotNull(method);
        return JsonSchemaGenerator.ForParameters(method.GetParameters());
    }

    private static MethodInfo GetMethod(Delegate function)
    {
        Guard.NotNull(function);
        return function.Method;
    }

    private static object? GetTarget(Delegate function)
    {
        Guard.NotNull(function);
        return function.Target;
    }

    private static async Task<object?> UnwrapResultAsync(object? returnValue)
    {
        if (returnValue is Task task)
        {
            await task.ConfigureAwait(false);
            var taskType = task.GetType();
            if (taskType.IsGenericType)
            {
                return taskType.GetProperty("Result")?.GetValue(task);
            }

            return null;
        }

        return returnValue;
    }
}
