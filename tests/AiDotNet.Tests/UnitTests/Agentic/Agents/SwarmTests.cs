using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Tools;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Agents
{
    public class SwarmTests
    {
        [Fact(Timeout = 60000)]
        public async Task RunAsync_EntryMemberAnswersDirectly_NoHandoff()
        {
            var triage = new SwarmMember<double>(
                "triage",
                ScriptedChatClient<double>.Sequence(ChatResponses.Text("Handled it.")),
                systemPrompt: "You are triage.");

            var swarm = new Swarm<double>(new[] { triage }, entryMemberName: "triage");

            var result = await swarm.RunAsync("Help me.");

            Assert.True(result.Completed);
            Assert.Equal("Handled it.", result.FinalText);
            Assert.Equal("triage", result.AgentName);
        }

        [Fact(Timeout = 60000)]
        public async Task RunAsync_HandoffTransfersControlToPeer_OverSharedHistory()
        {
            // Triage immediately hands off to the specialist; the specialist answers.
            var triage = new SwarmMember<double>(
                "triage",
                ScriptedChatClient<double>.Sequence(
                    ChatResponses.ToolCall("h1", "transfer_to_billing", "{}")),
                systemPrompt: "Route the user.");

            var billingClient = ScriptedChatClient<double>.Sequence(
                ChatResponses.Text("Your invoice is paid."));
            var billing = new SwarmMember<double>(
                "billing",
                billingClient,
                systemPrompt: "You handle billing.");

            var swarm = new Swarm<double>(new[] { triage, billing }, entryMemberName: "triage");

            var result = await swarm.RunAsync("Is my invoice paid?");

            Assert.True(result.Completed);
            Assert.Equal("Your invoice is paid.", result.FinalText);
            Assert.Equal("billing", result.AgentName);

            // The specialist saw the shared history, including the original user question.
            var billingRequest = billingClient.Requests[0];
            Assert.Contains(billingRequest, m => m.Role == ChatRole.User && m.Text == "Is my invoice paid?");
            // And the active member's own system prompt was applied.
            Assert.Equal(ChatRole.System, billingRequest[0].Role);
            Assert.Equal("You handle billing.", billingRequest[0].Text);
        }

        [Fact(Timeout = 60000)]
        public async Task RunAsync_ActiveMemberExecutesItsOwnTools()
        {
            var lookup = new RecordingTool("lookup", "Looks up a value.", _ => "found");
            var agentClient = new ScriptedChatClient<double>((callIndex, _) => callIndex == 0
                ? ChatResponses.ToolCall("t1", "lookup", "{}")
                : ChatResponses.Text("Done: found"));
            var solo = new SwarmMember<double>(
                "solo",
                agentClient,
                systemPrompt: "Use your tools.",
                tools: new ToolCollection().Add(lookup));

            var swarm = new Swarm<double>(new[] { solo }, entryMemberName: "solo");

            var result = await swarm.RunAsync("Look it up.");

            Assert.True(result.Completed);
            Assert.Single(lookup.Invocations);
            Assert.Equal("Done: found", result.FinalText);
        }

        [Fact(Timeout = 60000)]
        public async Task RunAsync_AdvertisesHandoffToolsForAllowedPeersOnly()
        {
            var a = new SwarmMember<double>(
                "a",
                ScriptedChatClient<double>.Sequence(ChatResponses.Text("ok")),
                handoffs: new[] { "b" }); // a may only transfer to b, not c
            var b = new SwarmMember<double>("b", ScriptedChatClient<double>.Sequence(ChatResponses.Text("b")));
            var c = new SwarmMember<double>("c", ScriptedChatClient<double>.Sequence(ChatResponses.Text("c")));

            var clientA = (ScriptedChatClient<double>)a.Client;
            var swarm = new Swarm<double>(new[] { a, b, c }, entryMemberName: "a");
            await swarm.RunAsync("hi");

            var options = clientA.ReceivedOptions[0];
            Assert.NotNull(options);
            Assert.NotNull(options.Tools);
            Assert.Contains(options.Tools, t => t.Name == "transfer_to_b");
            Assert.DoesNotContain(options.Tools, t => t.Name == "transfer_to_c");
        }

        [Fact(Timeout = 60000)]
        public async Task RunAsync_PingPongHandoffs_TerminateAtIterationCap()
        {
            // Two members that always hand off to each other -> would loop forever without the cap.
            var ping = new SwarmMember<double>(
                "ping",
                new ScriptedChatClient<double>((_, _) => ChatResponses.ToolCall("p", "transfer_to_pong", "{}")));
            var pong = new SwarmMember<double>(
                "pong",
                new ScriptedChatClient<double>((_, _) => ChatResponses.ToolCall("q", "transfer_to_ping", "{}")));

            var swarm = new Swarm<double>(new[] { ping, pong }, entryMemberName: "ping",
                new SwarmOptions { MaxIterations = 4 });

            var result = await swarm.RunAsync("go");

            Assert.False(result.Completed);
            Assert.Equal(4, result.Iterations);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_UnknownEntryMember_Throws()
        {
            var a = new SwarmMember<double>("a", ScriptedChatClient<double>.Sequence(ChatResponses.Text("x")));
            Assert.Throws<ArgumentException>(() =>
                new Swarm<double>(new[] { a }, entryMemberName: "missing"));
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_HandoffToUnknownPeer_Throws()
        {
            var a = new SwarmMember<double>("a",
                ScriptedChatClient<double>.Sequence(ChatResponses.Text("x")),
                handoffs: new[] { "ghost" });
            Assert.Throws<ArgumentException>(() =>
                new Swarm<double>(new[] { a }, entryMemberName: "a"));
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_DuplicateMemberNames_Throws()
        {
            var a1 = new SwarmMember<double>("dup", ScriptedChatClient<double>.Sequence(ChatResponses.Text("x")));
            var a2 = new SwarmMember<double>("dup", ScriptedChatClient<double>.Sequence(ChatResponses.Text("y")));
            Assert.Throws<ArgumentException>(() =>
                new Swarm<double>(new[] { a1, a2 }, entryMemberName: "dup"));
            await Task.CompletedTask;
        }
    }
}
