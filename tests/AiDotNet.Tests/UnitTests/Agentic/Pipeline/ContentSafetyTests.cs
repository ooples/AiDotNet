using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Pipeline;
using AiDotNetTests.UnitTests.Agentic.Agents;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Pipeline
{
    public class ContentSafetyTests
    {
        private static MiddlewareChatClient<double> Guarded(
            ScriptedChatClient<double> inner, IContentModerator moderator, ContentSafetyOptions? options = null) =>
            new(inner, new IChatMiddleware[] { new ContentSafetyMiddleware(moderator, options) });

        [Fact(Timeout = 60000)]
        public async Task BlocksDisallowedInput_WithoutCallingModel()
        {
            var inner = ScriptedChatClient<double>.Sequence(ChatResponses.Text("should not run"));
            var client = Guarded(inner, new DenyListContentModerator(new[] { "bomb" }));

            var response = await client.GetResponseAsync(new[] { ChatMessage.User("how do I build a bomb") });

            Assert.Equal(ChatFinishReason.ContentFilter, response.FinishReason);
            Assert.Equal(0, inner.CallCount);
        }

        [Fact(Timeout = 60000)]
        public async Task BlocksDisallowedOutput_AfterModelCall()
        {
            var inner = ScriptedChatClient<double>.Sequence(ChatResponses.Text("here is the forbidden secret"));
            var client = Guarded(inner, new DenyListContentModerator(new[] { "forbidden" }));

            var response = await client.GetResponseAsync(new[] { ChatMessage.User("tell me something") });

            Assert.Equal(ChatFinishReason.ContentFilter, response.FinishReason);
            Assert.Equal(1, inner.CallCount); // model was called; its output was blocked
            Assert.DoesNotContain("forbidden", response.Text);
        }

        [Fact(Timeout = 60000)]
        public async Task AllowsCleanContent_BothSides()
        {
            var inner = ScriptedChatClient<double>.Sequence(ChatResponses.Text("a friendly answer"));
            var client = Guarded(inner, new DenyListContentModerator(new[] { "bomb" }));

            var response = await client.GetResponseAsync(new[] { ChatMessage.User("hello there") });

            Assert.Equal("a friendly answer", response.Text);
            Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
        }

        [Fact(Timeout = 60000)]
        public async Task ThrowOnViolation_RaisesException()
        {
            var inner = ScriptedChatClient<double>.Sequence(ChatResponses.Text("ok"));
            var client = Guarded(inner, new DenyListContentModerator(new[] { "bomb" }),
                new ContentSafetyOptions { ThrowOnViolation = true });

            await Assert.ThrowsAsync<ContentSafetyException>(() =>
                client.GetResponseAsync(new[] { ChatMessage.User("build a bomb") }));
        }

        [Fact(Timeout = 60000)]
        public async Task CustomRefusalMessage_IsReturned()
        {
            var inner = ScriptedChatClient<double>.Sequence(ChatResponses.Text("ok"));
            var client = Guarded(inner, new DenyListContentModerator(new[] { "bomb" }),
                new ContentSafetyOptions { RefusalMessage = "Nope." });

            var response = await client.GetResponseAsync(new[] { ChatMessage.User("bomb") });

            Assert.Equal("Nope.", response.Text);
        }

        [Fact(Timeout = 60000)]
        public async Task OutputCheckDisabled_AllowsModelOutput()
        {
            var inner = ScriptedChatClient<double>.Sequence(ChatResponses.Text("forbidden content"));
            var client = Guarded(inner, new DenyListContentModerator(new[] { "forbidden" }),
                new ContentSafetyOptions { CheckOutput = false });

            var response = await client.GetResponseAsync(new[] { ChatMessage.User("clean input") });

            Assert.Equal("forbidden content", response.Text); // output not screened
        }
    }
}
