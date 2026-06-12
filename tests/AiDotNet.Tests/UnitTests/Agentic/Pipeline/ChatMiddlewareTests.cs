using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Pipeline;
using AiDotNetTests.UnitTests.Agentic.Agents;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Pipeline
{
    public class ChatMiddlewareTests
    {
        [Fact(Timeout = 60000)]
        public async Task Pipeline_RunsMiddlewareInRegistrationOrder()
        {
            await Task.Yield();

            var order = new List<string>();
            var inner = ScriptedChatClient<double>.Sequence(ChatResponses.Text("ok"));

            var a = new DelegateChatMiddleware(async (ctx, next, ct) =>
            {
                order.Add("a-before");
                var r = await next(ctx, ct);
                order.Add("a-after");
                return r;
            });
            var b = new DelegateChatMiddleware(async (ctx, next, ct) =>
            {
                order.Add("b-before");
                var r = await next(ctx, ct);
                order.Add("b-after");
                return r;
            });

            var client = new MiddlewareChatClient<double>(inner, new IChatMiddleware[] { a, b });
            await client.GetResponseAsync(new[] { ChatMessage.User("hi") });

            Assert.Equal(new[] { "a-before", "b-before", "b-after", "a-after" }, order);
        }

        [Fact(Timeout = 60000)]
        public async Task Middleware_CanRewriteRequest()
        {
            await Task.Yield();

            var inner = ScriptedChatClient<double>.Sequence(ChatResponses.Text("ok"));
            // Middleware prepends a system message the inner client then sees.
            var inject = new DelegateChatMiddleware((ctx, next, ct) =>
            {
                var messages = new List<ChatMessage> { ChatMessage.System("You are terse.") };
                messages.AddRange(ctx.Messages);
                ctx.Messages = messages;
                return next(ctx, ct);
            });

            var client = new MiddlewareChatClient<double>(inner, new IChatMiddleware[] { inject });
            await client.GetResponseAsync(new[] { ChatMessage.User("hi") });

            var seen = inner.Requests[0];
            Assert.Equal(ChatRole.System, seen[0].Role);
            Assert.Equal("You are terse.", seen[0].Text);
        }

        [Fact(Timeout = 60000)]
        public async Task Middleware_CanShortCircuit_WithoutCallingModel()
        {
            await Task.Yield();

            var inner = ScriptedChatClient<double>.Sequence(ChatResponses.Text("from model"));
            // A cache/guard that answers without hitting the model.
            var shortCircuit = new DelegateChatMiddleware((ctx, next, ct) =>
                Task.FromResult(new ChatResponse(ChatMessage.Assistant("cached"), ChatFinishReason.Stop)));

            var client = new MiddlewareChatClient<double>(inner, new IChatMiddleware[] { shortCircuit });
            var response = await client.GetResponseAsync(new[] { ChatMessage.User("hi") });

            Assert.Equal("cached", response.Text);
            Assert.Equal(0, inner.CallCount); // model never called
        }

        [Fact(Timeout = 60000)]
        public async Task Middleware_CanPostProcessResponse_AndShareState()
        {
            await Task.Yield();

            var inner = ScriptedChatClient<double>.Sequence(ChatResponses.Text("hello"));
            var post = new DelegateChatMiddleware(async (ctx, next, ct) =>
            {
                ctx.Items["tag"] = "seen";
                var r = await next(ctx, ct);
                return new ChatResponse(ChatMessage.Assistant(r.Text.ToUpperInvariant()), r.FinishReason, r.Usage, r.ModelId);
            });

            var client = new MiddlewareChatClient<double>(inner, new IChatMiddleware[] { post });
            var response = await client.GetResponseAsync(new[] { ChatMessage.User("hi") });

            Assert.Equal("HELLO", response.Text);
        }

        [Fact(Timeout = 60000)]
        public async Task NoMiddleware_DelegatesToInner()
        {
            await Task.Yield();

            var inner = ScriptedChatClient<double>.Sequence(ChatResponses.Text("passthrough"));
            var client = new MiddlewareChatClient<double>(inner, new List<IChatMiddleware>());

            var response = await client.GetResponseAsync(new[] { ChatMessage.User("hi") });

            Assert.Equal("passthrough", response.Text);
            Assert.Equal("scripted-test-client", client.ModelId);
        }
    }
}
