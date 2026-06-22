#if NET5_0_OR_GREATER
using System;
using System.Threading.Tasks;
using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Tools;
using AiDotNetTests.UnitTests.Agentic.Agents;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Hosting
{
    // The DI extensions live in AiDotNet.Serving (net10-only, where DI is referenced), so these tests are
    // net10-gated.
    public class AgenticDependencyInjectionTests
    {
        [Fact(Timeout = 60000)]
        public async Task AddAgentExecutor_ResolvesAgent_BuiltFromRegisteredClientAndTools()
        {
            await Task.Yield();

            var services = new ServiceCollection();
            services.AddAgentChatClient<double>(ScriptedChatClient<double>.Sequence(ChatResponses.Text("ok")));
            services.AddAgentTools(tools => tools.Add(new RecordingTool("noop", "Does nothing.", _ => "done")));
            services.AddAgentExecutor<double>(options => options.Name = "di-agent");

            using var provider = services.BuildServiceProvider();
            var agent = provider.GetRequiredService<AgentExecutor<double>>();

            Assert.Equal("di-agent", agent.Name);
            var result = await agent.RunAsync("hello");
            Assert.Equal("ok", result.FinalText);
        }

        [Fact(Timeout = 60000)]
        public async Task AddAgentExecutor_WorksWithoutTools()
        {
            await Task.Yield();

            var services = new ServiceCollection();
            services.AddAgentChatClient<double>(ScriptedChatClient<double>.Sequence(ChatResponses.Text("hi")));
            services.AddAgentExecutor<double>();

            using var provider = services.BuildServiceProvider();
            var agent = provider.GetRequiredService<AgentExecutor<double>>();

            Assert.Equal("hi", (await agent.RunAsync("x")).FinalText);
        }

        [Fact(Timeout = 60000)]
        public async Task AddAgentChatClient_Factory_IsResolved()
        {
            await Task.Yield();

            var services = new ServiceCollection();
            services.AddAgentChatClient<double>(_ => ScriptedChatClient<double>.Sequence(ChatResponses.Text("factory")));
            services.AddAgentExecutor<double>();

            using var provider = services.BuildServiceProvider();
            var client = provider.GetRequiredService<IChatClient<double>>();
            Assert.Equal("scripted-test-client", client.ModelId);

            var agent = provider.GetRequiredService<AgentExecutor<double>>();
            Assert.Equal("factory", (await agent.RunAsync("x")).FinalText);
        }

        [Fact(Timeout = 60000)]
        public async Task Extensions_GuardNullArguments()
        {
            await Task.Yield();

            var services = new ServiceCollection();
            Assert.Throws<ArgumentNullException>(() =>
                services.AddAgentChatClient<double>((IChatClient<double>)null));
            Assert.Throws<ArgumentNullException>(() => services.AddAgentTools(null));
        }
    }
}
#endif
