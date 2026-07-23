// This file intentionally passes null into non-nullable parameters to verify argument-null guards.
// Per the project guidelines, null-testing files disable nullable rather than using the null-forgiving
// operator (!), which would hide real problems.
#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Models
{
    public class ChatModelAbstractionTests
    {
        // ---- ChatMessage factories & projections ----

        [Fact(Timeout = 60000)]
        public async Task System_User_Assistant_Factories_SetRoleAndText()
        {
            await Task.Yield();

            Assert.Equal(ChatRole.System, ChatMessage.System("s").Role);
            Assert.Equal(ChatRole.User, ChatMessage.User("u").Role);
            Assert.Equal(ChatRole.Assistant, ChatMessage.Assistant("a").Role);
            Assert.Equal("u", ChatMessage.User("u").Text);
        }

        [Fact(Timeout = 60000)]
        public async Task Text_ConcatenatesOnlyTextParts_IgnoringNonText()
        {
            await Task.Yield();

            var msg = new ChatMessage(ChatRole.Assistant, new AiContent[]
            {
                new TextContent("Hello "),
                ImageContent.FromUri("https://example.com/a.png"),
                new TextContent("world"),
            });

            Assert.Equal("Hello world", msg.Text);
        }

        [Fact(Timeout = 60000)]
        public async Task ToolCalls_ExtractsOnlyToolCallParts()
        {
            await Task.Yield();

            var msg = ChatMessage.Assistant(new AiContent[]
            {
                new TextContent("let me check"),
                new ToolCallContent("call_1", "get_weather", "{\"city\":\"Paris\"}"),
                new ToolCallContent("call_2", "get_time"),
            });

            Assert.Equal(2, msg.ToolCalls.Count);
            Assert.Equal("get_weather", msg.ToolCalls[0].ToolName);
            Assert.Equal("call_2", msg.ToolCalls[1].CallId);
        }

        [Fact(Timeout = 60000)]
        public async Task Tool_Factory_ProducesToolRoleWithResultContent()
        {
            await Task.Yield();

            var msg = ChatMessage.Tool("call_1", "18C", isError: false);

            Assert.Equal(ChatRole.Tool, msg.Role);
            var result = Assert.IsType<ToolResultContent>(Assert.Single(msg.Contents));
            Assert.Equal("call_1", result.CallId);
            Assert.Equal("18C", result.Result);
            Assert.False(result.IsError);
        }

        [Fact(Timeout = 60000)]
        public async Task ChatMessage_Guards_NullContents_AndNullElements()
        {
            await Task.Yield();

            Assert.Throws<ArgumentNullException>(() =>
                new ChatMessage(ChatRole.User, (IReadOnlyList<AiContent>)null));

            Assert.Throws<ArgumentNullException>(() =>
                new ChatMessage(ChatRole.User, new AiContent[] { null }));
        }

        // ---- Content parts ----

        [Fact(Timeout = 60000)]
        public async Task TextContent_Guards_Null()
        {
            await Task.Yield();

            Assert.Throws<ArgumentNullException>(() => new TextContent(null));
        }

        [Fact(Timeout = 60000)]
        public async Task ImageContent_FromBytes_And_FromUri_SetExpectedState()
        {
            await Task.Yield();

            var bytes = new byte[] { 1, 2, 3 };
            var fromBytes = ImageContent.FromBytes(bytes, ImageMediaType.Png);
            Assert.True(fromBytes.HasData);
            Assert.Equal(ImageMediaType.Png, fromBytes.MediaType);
            Assert.Equal("image/png", fromBytes.MediaType?.ToMimeType());
            Assert.Null(fromBytes.Uri);

            var fromUri = ImageContent.FromUri("https://example.com/a.jpg", ImageMediaType.Jpeg);
            Assert.False(fromUri.HasData);
            Assert.Equal("https://example.com/a.jpg", fromUri.Uri);
            Assert.Equal(ImageMediaType.Jpeg, fromUri.MediaType);

            // Unspecified format when referenced by URI: null lets the provider infer it.
            Assert.Null(ImageContent.FromUri("https://example.com/a").MediaType);

            // MIME round-trip / parsing (incl. the common image/jpg alias).
            Assert.True(ImageMediaTypeExtensions.TryParseMimeType("image/jpg", out var parsed));
            Assert.Equal(ImageMediaType.Jpeg, parsed);
            Assert.False(ImageMediaTypeExtensions.TryParseMimeType("image/tiff", out _));
        }

        [Fact(Timeout = 60000)]
        public async Task ToolCallContent_DefaultsEmptyArguments_AndGuards()
        {
            await Task.Yield();

            Assert.Equal("{}", new ToolCallContent("id", "tool").ArgumentsJson);
            Assert.Equal("{}", new ToolCallContent("id", "tool", "   ").ArgumentsJson);
            Assert.Throws<ArgumentException>(() => new ToolCallContent("  ", "tool"));
            Assert.Throws<ArgumentException>(() => new ToolCallContent("id", "  "));
        }

        [Fact(Timeout = 60000)]
        public async Task ToolResultContent_TracksErrorFlag()
        {
            await Task.Yield();

            Assert.True(new ToolResultContent("id", "boom", isError: true).IsError);
            Assert.False(new ToolResultContent("id", "ok").IsError);
        }

        // ---- Tool definition / usage / response ----

        [Fact(Timeout = 60000)]
        public async Task AiToolDefinition_DefaultsToEmptyObjectSchema()
        {
            await Task.Yield();

            var def = new AiToolDefinition("get_weather", "Gets the weather");
            Assert.Equal("object", (string)def.ParametersSchema["type"]);
            Assert.NotNull(def.ParametersSchema["properties"]);

            var custom = new JObject { ["type"] = "object", ["required"] = new JArray("city") };
            Assert.Same(custom, new AiToolDefinition("t", "d", custom).ParametersSchema);

            Assert.Throws<ArgumentException>(() => new AiToolDefinition("  ", "d"));
        }

        [Fact(Timeout = 60000)]
        public async Task ChatUsage_TotalIsSum_AndGuardsNegative()
        {
            await Task.Yield();

            Assert.Equal(30, new ChatUsage(10, 20).TotalTokens);
            Assert.Throws<ArgumentOutOfRangeException>(() => new ChatUsage(-1, 0));
            Assert.Throws<ArgumentOutOfRangeException>(() => new ChatUsage(0, -1));
        }

        [Fact(Timeout = 60000)]
        public async Task ChatResponse_TextShortcut_AndGuardsNullMessage()
        {
            await Task.Yield();

            var resp = new ChatResponse(ChatMessage.Assistant("hi"), ChatFinishReason.Stop, new ChatUsage(1, 1), "m");
            Assert.Equal("hi", resp.Text);
            Assert.Equal(ChatFinishReason.Stop, resp.FinishReason);
            Assert.Equal("m", resp.ModelId);
            Assert.Throws<ArgumentNullException>(() => new ChatResponse(null));
        }

        [Fact(Timeout = 60000)]
        public async Task ChatResponseUpdate_Factories_SetExpectedFields()
        {
            await Task.Yield();

            Assert.Equal("hi", ChatResponseUpdate.ForText("hi").TextDelta);

            var tc = ChatResponseUpdate.ForToolCall(new StreamingToolCallUpdate(0, "id", "tool", "{"));
            Assert.Equal(0, tc.ToolCall.Index);
            Assert.Equal("tool", tc.ToolCall.ToolName);

            var fin = ChatResponseUpdate.ForFinish(ChatFinishReason.Stop, new ChatUsage(2, 3));
            Assert.Equal(ChatFinishReason.Stop, fin.FinishReason);
            Assert.Equal(5, fin.Usage.TotalTokens);

            Assert.Throws<ArgumentNullException>(() => ChatResponseUpdate.ForToolCall(null));
        }

        // ---- Interface round-trip against a fake client (exercises streaming on both TFMs) ----

        [Fact(Timeout = 60000)]
        public async Task FakeChatClient_NonStreaming_ReturnsResponse()
        {
            await Task.Yield();

            IChatClient<double> client = new FakeChatClient<double>();
            var resp = await client.GetResponseAsync(new[] { ChatMessage.User("hi") });

            Assert.Equal("fake", client.ModelId);
            Assert.Equal(ChatRole.Assistant, resp.Message.Role);
            Assert.Equal("echo: hi", resp.Text);
            Assert.Equal(ChatFinishReason.Stop, resp.FinishReason);
        }

        [Fact(Timeout = 60000)]
        public async Task FakeChatClient_Streaming_ReconstructsTextAndFinishes()
        {
            await Task.Yield();

            IChatClient<double> client = new FakeChatClient<double>();

            var text = new System.Text.StringBuilder();
            ChatFinishReason? finish = null;
            await foreach (var update in client.GetStreamingResponseAsync(new[] { ChatMessage.User("hi") }))
            {
                if (update.TextDelta != null) text.Append(update.TextDelta);
                if (update.FinishReason != null) finish = update.FinishReason;
            }

            Assert.Equal("echo: hi", text.ToString());
            Assert.Equal(ChatFinishReason.Stop, finish);
        }

        /// <summary>
        /// Minimal in-memory <see cref="IChatClient{T}"/> used to validate the abstraction's contract
        /// and the streaming pipeline without contacting any external provider.
        /// </summary>
        private sealed class FakeChatClient<T> : IChatClient<T>
        {
            public string ModelId => "fake";

            public Task<ChatResponse> GetResponseAsync(
                IReadOnlyList<ChatMessage> messages,
                ChatOptions options = null,
                CancellationToken cancellationToken = default)
            {
                var last = messages.Last().Text;
                var response = new ChatResponse(
                    ChatMessage.Assistant("echo: " + last),
                    ChatFinishReason.Stop,
                    new ChatUsage(1, 2),
                    ModelId);
                return Task.FromResult(response);
            }

            public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
                IReadOnlyList<ChatMessage> messages,
                ChatOptions options = null,
                [EnumeratorCancellation] CancellationToken cancellationToken = default)
            {
                var last = messages.Last().Text;
                yield return new ChatResponseUpdate(role: ChatRole.Assistant);
                await Task.Yield();
                yield return ChatResponseUpdate.ForText("echo: ");
                yield return ChatResponseUpdate.ForText(last);
                yield return ChatResponseUpdate.ForFinish(ChatFinishReason.Stop, new ChatUsage(1, 2));
            }
        }
    }
}
