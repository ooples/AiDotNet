using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Graph;
using AiDotNet.Agentic.Models;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Agents
{
    public class AgentGraphIntegrationTests
    {
        private sealed class FlowState
        {
            public string Topic { get; set; } = "";
            public string Draft { get; set; } = "";
            public string Final { get; set; } = "";
        }

        private static AgentExecutor<double> Agent(string name, string answer) =>
            new(ScriptedChatClient<double>.Sequence(ChatResponses.Text(answer)),
                tools: null, new AgentExecutorOptions { Name = name });

        [Fact(Timeout = 60000)]
        public async Task TwoAgentNodes_RunInSequence_OverTypedState()
        {
            var graph = new StateGraph<FlowState>();
            graph.AddAgentNode("research", Agent("researcher", "facts about cats"),
                state => "Research: " + state.Topic,
                (state, result) => new FlowState { Topic = state.Topic, Draft = result.FinalText, Final = state.Final });
            graph.AddAgentNode("write", Agent("writer", "A polished article."),
                state => "Write from: " + state.Draft,
                (state, result) => new FlowState { Topic = state.Topic, Draft = state.Draft, Final = result.FinalText });
            graph.AddEdge("research", "write");
            graph.AddEdge("write", StateGraph<FlowState>.End);
            graph.SetEntryPoint("research");

            var compiled = graph.Compile();
            var result = await compiled.InvokeAsync(new FlowState { Topic = "cats" });

            Assert.Equal("facts about cats", result.Draft); // researcher node ran
            Assert.Equal("A polished article.", result.Final); // writer node ran after
        }

        [Fact(Timeout = 60000)]
        public async Task ConditionalEdge_RoutesBetweenAgentNodes()
        {
            var graph = new StateGraph<FlowState>();
            graph.AddAgentNode("classify", Agent("classifier", "cats"),
                state => state.Topic,
                (state, result) => new FlowState { Topic = result.FinalText, Draft = state.Draft, Final = state.Final });
            graph.AddAgentNode("cats", Agent("cat-expert", "meow facts"),
                state => state.Topic,
                (state, result) => new FlowState { Topic = state.Topic, Draft = state.Draft, Final = result.FinalText });
            graph.AddAgentNode("dogs", Agent("dog-expert", "woof facts"),
                state => state.Topic,
                (state, result) => new FlowState { Topic = state.Topic, Draft = state.Draft, Final = result.FinalText });

            graph.AddConditionalEdges("classify", state => state.Topic == "cats" ? "cats" : "dogs");
            graph.AddEdge("cats", StateGraph<FlowState>.End);
            graph.AddEdge("dogs", StateGraph<FlowState>.End);
            graph.SetEntryPoint("classify");

            var result = await graph.Compile().InvokeAsync(new FlowState { Topic = "anything" });

            // classify -> "cats" -> routed to the cat expert.
            Assert.Equal("meow facts", result.Final);
        }

        [Fact(Timeout = 60000)]
        public async Task SupervisorAgent_AsAGraphNode()
        {
            // A whole supervisor team is a single graph node.
            var worker = Agent("calc", "42");
            var coordinator = ScriptedChatClient<double>.Sequence(
                ChatResponses.ToolCall("c1", "transfer_to_calc", "{\"task\":\"compute\"}"),
                ChatResponses.Text("The answer is 42."));
            var supervisor = new SupervisorAgent<double>(coordinator, new IAgent<double>[] { worker });

            var graph = new StateGraph<FlowState>();
            graph.AddAgentNode("team", supervisor,
                state => state.Topic,
                (state, result) => new FlowState { Topic = state.Topic, Draft = state.Draft, Final = result.FinalText });
            graph.AddEdge("team", StateGraph<FlowState>.End);
            graph.SetEntryPoint("team");

            var result = await graph.Compile().InvokeAsync(new FlowState { Topic = "what is 6x7?" });

            Assert.Equal("The answer is 42.", result.Final);
        }
    }
}
