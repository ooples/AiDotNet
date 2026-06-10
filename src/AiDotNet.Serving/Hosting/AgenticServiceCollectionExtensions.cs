using System;
using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Tools;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>
/// Dependency-injection registration helpers for the AiDotNet agentic stack — the hosting integration that
/// lets agents, their chat client, and their tools be composed through <see cref="IServiceCollection"/>
/// (Generic Host / ASP.NET / .NET Aspire), parallel to Semantic Kernel's <c>AddKernel</c>-style extensions.
/// </summary>
/// <remarks>
/// <para>
/// These live in the opt-in <c>AiDotNet.Serving</c> package so the core library carries no DI dependency.
/// Register a chat client and (optionally) a tool collection, then resolve an <see cref="AgentExecutor{T}"/>
/// built from them.
/// </para>
/// <para><b>For Beginners:</b> If your app uses .NET's built-in dependency injection, these one-liners wire
/// up the model client, the tools, and the agent so you can ask the container for a ready-to-use agent.
/// </para>
/// </remarks>
public static class AgenticServiceCollectionExtensions
{
    /// <summary>
    /// Registers a chat client instance as a singleton <see cref="IChatClient{T}"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of the client.</typeparam>
    /// <param name="services">The service collection.</param>
    /// <param name="client">The chat client to register.</param>
    /// <returns>The service collection, for chaining.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="services"/> or <paramref name="client"/> is <c>null</c>.</exception>
    public static IServiceCollection AddAgentChatClient<T>(this IServiceCollection services, IChatClient<T> client)
    {
        if (services is null)
        {
            throw new ArgumentNullException(nameof(services));
        }

        if (client is null)
        {
            throw new ArgumentNullException(nameof(client));
        }

        return services.AddSingleton(client);
    }

    /// <summary>
    /// Registers a chat client via a factory as a singleton <see cref="IChatClient{T}"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of the client.</typeparam>
    /// <param name="services">The service collection.</param>
    /// <param name="factory">Builds the chat client from the service provider.</param>
    /// <returns>The service collection, for chaining.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="services"/> or <paramref name="factory"/> is <c>null</c>.</exception>
    public static IServiceCollection AddAgentChatClient<T>(this IServiceCollection services, Func<IServiceProvider, IChatClient<T>> factory)
    {
        if (services is null)
        {
            throw new ArgumentNullException(nameof(services));
        }

        if (factory is null)
        {
            throw new ArgumentNullException(nameof(factory));
        }

        return services.AddSingleton(factory);
    }

    /// <summary>
    /// Builds and registers a singleton <see cref="ToolCollection"/> the agents may use.
    /// </summary>
    /// <param name="services">The service collection.</param>
    /// <param name="configure">Configures the tool collection.</param>
    /// <returns>The service collection, for chaining.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="services"/> or <paramref name="configure"/> is <c>null</c>.</exception>
    public static IServiceCollection AddAgentTools(this IServiceCollection services, Action<ToolCollection> configure)
    {
        if (services is null)
        {
            throw new ArgumentNullException(nameof(services));
        }

        if (configure is null)
        {
            throw new ArgumentNullException(nameof(configure));
        }

        var tools = new ToolCollection();
        configure(tools);
        return services.AddSingleton(tools);
    }

    /// <summary>
    /// Registers an <see cref="AgentExecutor{T}"/> built from the registered <see cref="IChatClient{T}"/> and
    /// (if present) a registered <see cref="ToolCollection"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of the agent.</typeparam>
    /// <param name="services">The service collection.</param>
    /// <param name="configure">Optionally configures the executor options.</param>
    /// <returns>The service collection, for chaining.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="services"/> is <c>null</c>.</exception>
    public static IServiceCollection AddAgentExecutor<T>(this IServiceCollection services, Action<AgentExecutorOptions>? configure = null)
    {
        if (services is null)
        {
            throw new ArgumentNullException(nameof(services));
        }

        return services.AddSingleton(serviceProvider =>
        {
            var client = serviceProvider.GetRequiredService<IChatClient<T>>();
            var tools = serviceProvider.GetService<ToolCollection>();
            var options = new AgentExecutorOptions();
            configure?.Invoke(options);
            return new AgentExecutor<T>(client, tools, options);
        });
    }
}
