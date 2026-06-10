using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Models.Connectors;
using Xunit;
using Meai = Microsoft.Extensions.AI;

namespace AiDotNetTests.UnitTests.Agentic.Connectors
{
    public class MeaiChatClientTests
    {
        // ---- Inbound: MEAI model -> AiDotNet IChatClient<T> ----

        /// <summary>Fake MEAI client: echoes captured options and optionally emits a tool call when tools are supplied.</summary>
        private sealed class FakeMeaiClient : Meai.IChatClient
        {
            public Meai.ChatOptions? LastOptions { get; private set; }

            public List<Meai.ChatMessage> LastMessages { get; private set; } = new();

            public Task<Meai.ChatResponse> GetResponseAsync(
                IEnumerable<Meai.ChatMessage> messages,
                Meai.ChatOptions? options = null,
                CancellationToken cancellationToken = default)
            {
                LastOptions = options;
                LastMessages = messages.ToList();

                if (options?.Tools is { Count: > 0 } tools && tools[0] is Meai.AIFunction function)
                {
                    var call = new Meai.FunctionCallContent(
                        "call-1", function.Name, new Dictionary<string, object?> { ["city"] = "Paris" });
                    var toolMessage = new Meai.ChatMessage(Meai.ChatRole.Assistant, new List<Meai.AIContent> { call });
                    return Task.FromResult(new Meai.ChatResponse(toolMessage)
                    {
                        FinishReason = Meai.ChatFinishReason.ToolCalls,
                        ModelId = "fake-meai-model",
                    });
                }

                var response = new Meai.ChatResponse(new Meai.ChatMessage(Meai.ChatRole.Assistant, "hi from meai"))
                {
                    FinishReason = Meai.ChatFinishReason.Stop,
                    ModelId = "fake-meai-model",
                    Usage = new Meai.UsageDetails { InputTokenCount = 3, OutputTokenCount = 4 }
                };
                return Task.FromResult(response);
            }

            public async IAsyncEnumerable<Meai.ChatResponseUpdate> GetStreamingResponseAsync(
                IEnumerable<Meai.ChatMessage> messages,
                Meai.ChatOptions? options = null,
                [EnumeratorCancellation] CancellationToken cancellationToken = default)
            {
                await Task.CompletedTask;
                yield return new Meai.ChatResponseUpdate(Meai.ChatRole.Assistant, "hi ");
                yield return new Meai.ChatResponseUpdate(Meai.ChatRole.Assistant, "from meai")
                {
                    FinishReason = Meai.ChatFinishReason.Stop
                };
            }

            public object? GetService(Type serviceType, object? serviceKey = null) => null;

            public void Dispose() { }
        }

        [Fact(Timeout = 60000)]
        public async Task WrapsMeaiClient_MapsResponseTextFinishAndUsage()
        {
            IChatClient<double> client = new MeaiChatClient<double>(new FakeMeaiClient(), "fake-meai-model");

            var response = await client.GetResponseAsync(new[] { ChatMessage.User("hi") });

            Assert.Equal("hi from meai", response.Text);
            Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
            Assert.Equal("fake-meai-model", response.ModelId);
            Assert.Equal(7, response.Usage?.TotalTokens);
        }

        [Fact(Timeout = 60000)]
        public async Task Streaming_ReconstructsText()
        {
            IChatClient<double> client = new MeaiChatClient<double>(new FakeMeaiClient());

            var text = new System.Text.StringBuilder();
            await foreach (var update in client.GetStreamingResponseAsync(new[] { ChatMessage.User("hi") }))
            {
                if (update.TextDelta != null) text.Append(update.TextDelta);
            }

            Assert.Equal("hi from meai", text.ToString());
        }

        [Fact(Timeout = 60000)]
        public async Task Inbound_PassesToolsThrough_AndSurfacesToolCalls()
        {
            var fake = new FakeMeaiClient();
            IChatClient<double> client = new MeaiChatClient<double>(fake);
            var options = new ChatOptions
            {
                Tools = new[] { new AiToolDefinition("get_weather", "Gets the weather for a city.") },
                ToolChoice = ToolChoiceMode.Required,
                RequiredToolName = "get_weather",
            };

            var response = await client.GetResponseAsync(new[] { ChatMessage.User("weather in Paris?") }, options);

            // The MEAI model received the tool declaration and the required-specific tool mode.
            var capturedTools = fake.LastOptions?.Tools;
            Assert.NotNull(capturedTools);
            Assert.Equal("get_weather", Assert.IsAssignableFrom<Meai.AIFunction>(capturedTools[0]).Name);
            Assert.IsType<Meai.RequiredChatToolMode>(fake.LastOptions?.ToolMode);

            // The model's tool call surfaced as an AiDotNet ToolCallContent with the ToolCalls finish reason.
            Assert.Equal(ChatFinishReason.ToolCalls, response.FinishReason);
            var call = Assert.Single(response.Message.ToolCalls);
            Assert.Equal("get_weather", call.ToolName);
            Assert.Equal("call-1", call.CallId);
            Assert.Contains("Paris", call.ArgumentsJson);
        }

        [Fact(Timeout = 60000)]
        public async Task Inbound_ReplaysToolResults_AsMeaiFunctionResults()
        {
            var fake = new FakeMeaiClient();
            IChatClient<double> client = new MeaiChatClient<double>(fake);

            var messages = new[]
            {
                ChatMessage.User("weather?"),
                ChatMessage.Assistant(new AiContent[] { new ToolCallContent("call-1", "get_weather", "{\"city\":\"Paris\"}") }),
                ChatMessage.Tool("call-1", "18C and sunny"),
            };

            await client.GetResponseAsync(messages);

            // The tool-call turn mapped to a MEAI FunctionCallContent, and the tool-result turn to a
            // FunctionResultContent — the full tool round-trip is replayable back to the MEAI model.
            var allContents = fake.LastMessages.SelectMany(m => m.Contents).ToList();
            var call = allContents.OfType<Meai.FunctionCallContent>().Single();
            Assert.Equal("get_weather", call.Name);
            Assert.Equal("call-1", call.CallId);
            Assert.Contains("Paris", MeaiInterop.ArgumentsToJson(call.Arguments));

            var result = allContents.OfType<Meai.FunctionResultContent>().Single();
            Assert.Equal("call-1", result.CallId);
            Assert.Equal("18C and sunny", result.Result?.ToString());
        }

        // ---- Outbound: AiDotNet IChatClient<T> -> MEAI IChatClient ----

        /// <summary>Fake AiDotNet client: echoes captured options and emits a tool call when tools are supplied.</summary>
        private sealed class FakeAgenticClient : IChatClient<double>
        {
            public ChatOptions? LastOptions { get; private set; }

            public string ModelId => "fake-agentic";

            public Task<ChatResponse> GetResponseAsync(
                IReadOnlyList<ChatMessage> messages,
                ChatOptions? options = null,
                CancellationToken cancellationToken = default)
            {
                LastOptions = options;

                if (options?.Tools is { Count: > 0 } tools)
                {
                    var msg = ChatMessage.Assistant(new AiContent[]
                    {
                        new ToolCallContent("call-9", tools[0].Name, "{\"city\":\"Paris\"}")
                    });
                    return Task.FromResult(new ChatResponse(msg, ChatFinishReason.ToolCalls, new ChatUsage(2, 3), "fake-agentic"));
                }

                return Task.FromResult(new ChatResponse(
                    ChatMessage.Assistant("hi from aidotnet"), ChatFinishReason.Stop, new ChatUsage(1, 1), "fake-agentic"));
            }

            public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
                IReadOnlyList<ChatMessage> messages,
                ChatOptions? options = null,
                [EnumeratorCancellation] CancellationToken cancellationToken = default)
            {
                await Task.CompletedTask;
                yield return ChatResponseUpdate.ForText("hi ");
                yield return ChatResponseUpdate.ForText("there");
                yield return ChatResponseUpdate.ForFinish(ChatFinishReason.Stop);
            }
        }

        /// <summary>Minimal schema-only MEAI tool for driving the outbound adapter.</summary>
        private sealed class TestMeaiFunction : Meai.AIFunction
        {
            public TestMeaiFunction(string name, string description)
            {
                Name = name;
                Description = description;
            }

            public override string Name { get; }

            public override string Description { get; }

            protected override ValueTask<object?> InvokeCoreAsync(
                Meai.AIFunctionArguments arguments, CancellationToken cancellationToken) =>
                throw new NotSupportedException("test declaration");
        }

        [Fact(Timeout = 60000)]
        public async Task Outbound_MapsTextFinishAndUsage()
        {
            Meai.IChatClient meai = new FakeAgenticClient().AsMeaiChatClient();

            var response = await meai.GetResponseAsync(new[] { new Meai.ChatMessage(Meai.ChatRole.User, "hi") });

            Assert.Equal("hi from aidotnet", response.Text);
            Assert.Equal(Meai.ChatFinishReason.Stop, response.FinishReason);
            Assert.Equal("fake-agentic", response.ModelId);
        }

        [Fact(Timeout = 60000)]
        public async Task Outbound_Streaming_ReconstructsText()
        {
            Meai.IChatClient meai = new FakeAgenticClient().AsMeaiChatClient();

            var text = new System.Text.StringBuilder();
            await foreach (var update in meai.GetStreamingResponseAsync(new[] { new Meai.ChatMessage(Meai.ChatRole.User, "hi") }))
            {
                text.Append(update.Text);
            }

            Assert.Equal("hi there", text.ToString());
        }

        [Fact(Timeout = 60000)]
        public async Task Outbound_PassesToolsThrough_AndEmitsFunctionCall()
        {
            var fake = new FakeAgenticClient();
            Meai.IChatClient meai = fake.AsMeaiChatClient();
            var options = new Meai.ChatOptions
            {
                Tools = new List<Meai.AITool> { new TestMeaiFunction("get_weather", "Gets the weather.") },
            };

            var response = await meai.GetResponseAsync(
                new[] { new Meai.ChatMessage(Meai.ChatRole.User, "weather?") }, options);

            // The AiDotNet client received the tool declaration.
            var capturedTools = fake.LastOptions?.Tools;
            Assert.NotNull(capturedTools);
            Assert.Equal("get_weather", capturedTools[0].Name);

            // The AiDotNet tool call surfaced as a MEAI FunctionCallContent.
            var functionCall = response.Messages
                .SelectMany(m => m.Contents)
                .OfType<Meai.FunctionCallContent>()
                .Single();
            Assert.Equal("get_weather", functionCall.Name);
            Assert.Equal("call-9", functionCall.CallId);
            Assert.Equal(Meai.ChatFinishReason.ToolCalls, response.FinishReason);
        }

        [Fact(Timeout = 60000)]
        public async Task RoundTrip_AgenticToMeaiAndBack_PreservesText()
        {
            // AiDotNet client -> expose as MEAI -> wrap that back as AiDotNet: text must survive both hops.
            Meai.IChatClient asMeai = new FakeAgenticClient().AsMeaiChatClient();
            IChatClient<double> roundTripped = asMeai.AsAgenticChatClient<double>("round-trip");

            var response = await roundTripped.GetResponseAsync(new[] { ChatMessage.User("hi") });

            Assert.Equal("hi from aidotnet", response.Text);
            Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
        }
    }
}
