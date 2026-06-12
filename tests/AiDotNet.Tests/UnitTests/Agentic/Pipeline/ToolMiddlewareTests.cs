using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Pipeline;
using AiDotNet.Agentic.Tools;
using AiDotNetTests.UnitTests.Agentic.Agents;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Pipeline
{
    public class ToolMiddlewareTests
    {
        private static RecordingTool Tool() =>
            new("act", "Performs an action.", args => "done:" + (string?)args["x"]);

        [Fact(Timeout = 60000)]
        public async Task ApprovalMiddleware_AllowsApprovedCall()
        {
            await Task.Yield();

            var tool = Tool();
            var wrapped = new MiddlewareAgentTool(tool, new IToolMiddleware[]
            {
                new ApprovalToolMiddleware(_ => true),
            });

            var result = await wrapped.InvokeAsync(new JObject { ["x"] = "1" });

            Assert.False(result.IsError);
            Assert.Equal("done:1", result.Content);
            Assert.Single(tool.Invocations);
        }

        [Fact(Timeout = 60000)]
        public async Task ApprovalMiddleware_DeniesUnapprovedCall_WithoutRunningTool()
        {
            await Task.Yield();

            var tool = Tool();
            var wrapped = new MiddlewareAgentTool(tool, new IToolMiddleware[]
            {
                new ApprovalToolMiddleware(_ => false, "Not allowed."),
            });

            var result = await wrapped.InvokeAsync(new JObject { ["x"] = "1" });

            Assert.True(result.IsError);
            Assert.Equal("Not allowed.", result.Content);
            Assert.Empty(tool.Invocations); // tool never ran
        }

        [Fact(Timeout = 60000)]
        public async Task Middleware_RunsInOrder_AndCanRewriteArguments()
        {
            await Task.Yield();

            var order = new List<string>();
            var tool = Tool();
            var wrapped = new MiddlewareAgentTool(tool, new IToolMiddleware[]
            {
                new DelegateToolMiddleware(async (ctx, next, ct) =>
                {
                    order.Add("outer");
                    ctx.Arguments["x"] = "rewritten";
                    return await next(ctx, ct);
                }),
                new DelegateToolMiddleware(async (ctx, next, ct) =>
                {
                    order.Add("inner");
                    return await next(ctx, ct);
                }),
            });

            var result = await wrapped.InvokeAsync(new JObject { ["x"] = "original" });

            Assert.Equal(new[] { "outer", "inner" }, order);
            Assert.Equal("done:rewritten", result.Content);
            Assert.Equal("rewritten", (string?)tool.Invocations[0]["x"]);
        }

        [Fact(Timeout = 60000)]
        public async Task WrappedTool_KeepsInnerIdentity()
        {
            await Task.Yield();

            var wrapped = new MiddlewareAgentTool(Tool(), new IToolMiddleware[] { new ApprovalToolMiddleware(_ => true) });
            Assert.Equal("act", wrapped.Name);
            Assert.Equal("Performs an action.", wrapped.Description);
            Assert.Equal("act", wrapped.ToDefinition().Name);
        }

        [Fact(Timeout = 60000)]
        public async Task IntegratesWithAgentExecutor_ApprovalBlocksToolMidLoop()
        {
            await Task.Yield();

            // The model calls "act"; an approval gate denies it; the model then answers with the deny result.
            var tool = Tool();
            var gated = new MiddlewareAgentTool(tool, new IToolMiddleware[]
            {
                new ApprovalToolMiddleware(_ => false, "denied"),
            });
            var tools = new ToolCollection().Add(gated);

            var client = ScriptedChatClient<double>.Sequence(
                ChatResponses.ToolCall("c1", "act", "{\"x\":\"1\"}"),
                ChatResponses.Text("Understood, it was denied."));
            var agent = new AgentExecutor<double>(client, tools);

            var run = await agent.RunAsync("do the action");

            Assert.True(run.Completed);
            Assert.Empty(tool.Invocations); // gate blocked execution
            // The model's second turn saw the deny message as the tool result.
            var secondRequest = client.Requests[1];
            Assert.Contains(secondRequest, m => m.Role == ChatRole.Tool
                && m.Contents.OfType<ToolResultContent>().Any(r => r.Result == "denied" && r.IsError));
        }
    }
}
