using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Tools;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Agents
{
    public class AgentExecutorTests
    {
        [Fact(Timeout = 60000)]
        public async Task RunAsync_PlainAnswer_ReturnsTextInOneIteration()
        {
            await Task.Yield();

            var client = ScriptedChatClient<double>.Sequence(ChatResponses.Text("The answer is 42."));
            var agent = new AgentExecutor<double>(client);

            var result = await agent.RunAsync("What is the answer?");

            Assert.True(result.Completed);
            Assert.Equal(1, result.Iterations);
            Assert.Equal("The answer is 42.", result.FinalText);
            Assert.Equal(1, client.CallCount);
        }

        [Fact(Timeout = 60000)]
        public async Task RunAsync_ToolCall_RunsToolThenFeedsResultBack()
        {
            await Task.Yield();

            var tool = new RecordingTool("add", "Adds two numbers.", args =>
            {
                var a = (int)args["a"];
                var b = (int)args["b"];
                return (a + b).ToString();
            });
            var tools = new ToolCollection().Add(tool);

            // Call 0: model asks to use the tool. Call 1: model gives the final answer.
            var client = ScriptedChatClient<double>.Sequence(
                ChatResponses.ToolCall("call-1", "add", "{\"a\":2,\"b\":3}"),
                ChatResponses.Text("The sum is 5."));
            var agent = new AgentExecutor<double>(client, tools);

            var result = await agent.RunAsync("Add 2 and 3.");

            Assert.True(result.Completed);
            Assert.Equal(2, result.Iterations);
            Assert.Equal("The sum is 5.", result.FinalText);

            // The tool was invoked with the model's arguments.
            Assert.Single(tool.Invocations);
            Assert.Equal(2, (int)tool.Invocations[0]["a"]);
            Assert.Equal(3, (int)tool.Invocations[0]["b"]);

            // The second model call saw the tool-result message appended to the transcript.
            var secondRequest = client.Requests[1];
            Assert.Contains(secondRequest, m => m.Role == ChatRole.Tool && m.Contents.OfType<ToolResultContent>().Any(r => r.Result == "5"));
        }

        [Fact(Timeout = 60000)]
        public async Task RunAsync_SystemPrompt_IsPrependedToTheConversation()
        {
            await Task.Yield();

            var client = ScriptedChatClient<double>.Sequence(ChatResponses.Text("ok"));
            var options = new AgentExecutorOptions { SystemPrompt = "You are a pirate." };
            var agent = new AgentExecutor<double>(client, tools: null, options);

            await agent.RunAsync("Hello");

            var firstRequest = client.Requests[0];
            Assert.Equal(ChatRole.System, firstRequest[0].Role);
            Assert.Equal("You are a pirate.", firstRequest[0].Text);
            Assert.Equal(ChatRole.User, firstRequest[1].Role);
        }

        [Fact(Timeout = 60000)]
        public async Task RunAsync_NoTools_DoesNotAdvertiseToolsToTheModel()
        {
            await Task.Yield();

            var client = ScriptedChatClient<double>.Sequence(ChatResponses.Text("ok"));
            var agent = new AgentExecutor<double>(client);

            await agent.RunAsync("Hello");

            var options = client.ReceivedOptions[0];
            Assert.NotNull(options);
            Assert.True(options.Tools is null || options.Tools.Count == 0);
            Assert.Null(options.ToolChoice);
        }

        [Fact(Timeout = 60000)]
        public async Task RunAsync_WithTools_AdvertisesToolsAndAutoChoice()
        {
            await Task.Yield();

            var tools = new ToolCollection().Add(new RecordingTool("noop", "Does nothing.", _ => "done"));
            var client = ScriptedChatClient<double>.Sequence(ChatResponses.Text("ok"));
            var agent = new AgentExecutor<double>(client, tools);

            await agent.RunAsync("Hello");

            var options = client.ReceivedOptions[0];
            Assert.NotNull(options);
            Assert.NotNull(options.Tools);
            Assert.Contains(options.Tools, d => d.Name == "noop");
            Assert.Equal(ToolChoiceMode.Auto, options.ToolChoice);
        }

        [Fact(Timeout = 60000)]
        public async Task RunAsync_HitsIterationCap_StopsWithCompletedFalse()
        {
            await Task.Yield();

            var tools = new ToolCollection().Add(new RecordingTool("loop", "Loops forever.", _ => "again"));
            // The model always asks for the tool again, never producing a final answer.
            var client = new ScriptedChatClient<double>((_, _) =>
                ChatResponses.ToolCall("c", "loop", "{}"));
            var options = new AgentExecutorOptions { MaxIterations = 3 };
            var agent = new AgentExecutor<double>(client, tools, options);

            var result = await agent.RunAsync("Go");

            Assert.False(result.Completed);
            Assert.Equal(3, result.Iterations);
            Assert.Equal(3, client.CallCount);
        }

        [Fact(Timeout = 60000)]
        public async Task RunAsync_AggregatesUsageAcrossCalls()
        {
            await Task.Yield();

            var tools = new ToolCollection().Add(new RecordingTool("add", "Adds.", _ => "5"));
            var client = ScriptedChatClient<double>.Sequence(
                ChatResponses.ToolCall("call-1", "add", "{}", new ChatUsage(10, 4)),
                ChatResponses.Text("done", new ChatUsage(7, 3)));
            var agent = new AgentExecutor<double>(client, tools);

            var result = await agent.RunAsync("Add");

            Assert.NotNull(result.Usage);
            Assert.Equal(17, result.Usage.InputTokens);
            Assert.Equal(7, result.Usage.OutputTokens);
            Assert.Equal(24, result.Usage.TotalTokens);
        }

        [Fact(Timeout = 60000)]
        public async Task RunAsync_EmptyConversation_Throws()
        {
            await Task.Yield();

            var client = ScriptedChatClient<double>.Sequence(ChatResponses.Text("ok"));
            var agent = new AgentExecutor<double>(client);

            await Assert.ThrowsAsync<ArgumentException>(() =>
                agent.RunAsync(new List<ChatMessage>()));
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_NullClient_Throws()
        {
            await Task.Yield();

            Assert.Throws<ArgumentNullException>(() => new AgentExecutor<double>(client: null));
        }

        [Fact(Timeout = 60000)]
        public async Task RunAsync_ExposesNameAndDescriptionFromOptions()
        {
            await Task.Yield();

            var client = ScriptedChatClient<double>.Sequence(ChatResponses.Text("ok"));
            var options = new AgentExecutorOptions { Name = "researcher", Description = "Finds facts." };
            var agent = new AgentExecutor<double>(client, tools: null, options);

            Assert.Equal("researcher", agent.Name);
            Assert.Equal("Finds facts.", agent.Description);

            // Default name when unset.
            var bare = new AgentExecutor<double>(client);
            Assert.Equal("agent", bare.Name);

            await agent.RunAsync("hi");
        }
    }
}
