using System.Collections.Generic;
using AiDotNet.Serving.Engine.Http;
using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>Tests for the default chat template that renders OpenAI chat messages into a model prompt.</summary>
public class ChatTemplateTests
{
    [Fact]
    public void DefaultTemplate_RendersRolesAndOpensAssistantTurn()
    {
        var template = new DefaultChatTemplate();
        var prompt = template.Render(new List<ChatMessage>
        {
            new() { Role = "system", Content = "be nice" },
            new() { Role = "user", Content = "hello" },
        });

        Assert.Equal("<|system|>\nbe nice\n<|user|>\nhello\n<|assistant|>\n", prompt);
    }

    [Fact]
    public void DefaultTemplate_EmptyConversation_StillOpensAssistantTurn()
    {
        var prompt = new DefaultChatTemplate().Render(new List<ChatMessage>());
        Assert.Equal("<|assistant|>\n", prompt);
    }

    [Fact]
    public void DefaultTemplate_BlankRole_DefaultsToUser()
    {
        var prompt = new DefaultChatTemplate().Render(new List<ChatMessage> { new() { Role = "", Content = "x" } });
        Assert.Equal("<|user|>\nx\n<|assistant|>\n", prompt);
    }
}
