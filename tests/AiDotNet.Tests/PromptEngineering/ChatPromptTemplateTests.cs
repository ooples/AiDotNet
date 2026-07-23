using AiDotNet.PromptEngineering.Templates;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.PromptEngineering;

public class ChatPromptTemplateTests
{
    [Fact(Timeout = 60000)]
    public async Task Constructor_CreatesEmptyTemplate()
    {
        var template = new ChatPromptTemplate();

        Assert.NotNull(template);
        Assert.Empty(template.Messages);
    }

    [Fact(Timeout = 60000)]
    public async Task AddSystemMessage_AddsMessageCorrectly()
    {
        var template = new ChatPromptTemplate();

        template.AddSystemMessage("You are a helpful assistant");

        Assert.Single(template.Messages);
        Assert.Equal("system", template.Messages[0].Role);
        Assert.Equal("You are a helpful assistant", template.Messages[0].Content);
    }

    [Fact(Timeout = 60000)]
    public async Task AddUserMessage_AddsMessageCorrectly()
    {
        var template = new ChatPromptTemplate();

        template.AddUserMessage("What is the weather?");

        Assert.Single(template.Messages);
        Assert.Equal("user", template.Messages[0].Role);
        Assert.Equal("What is the weather?", template.Messages[0].Content);
    }

    [Fact(Timeout = 60000)]
    public async Task AddAssistantMessage_AddsMessageCorrectly()
    {
        var template = new ChatPromptTemplate();

        template.AddAssistantMessage("It's sunny today.");

        Assert.Single(template.Messages);
        Assert.Equal("assistant", template.Messages[0].Role);
        Assert.Equal("It's sunny today.", template.Messages[0].Content);
    }

    [Fact(Timeout = 60000)]
    public async Task AddMessage_WithCustomRole_AddsMessageCorrectly()
    {
        var template = new ChatPromptTemplate();

        template.AddMessage("custom", "Custom message");

        Assert.Single(template.Messages);
        Assert.Equal("custom", template.Messages[0].Role);
        Assert.Equal("Custom message", template.Messages[0].Content);
    }

    [Fact(Timeout = 60000)]
    public async Task AddMultipleMessages_CreatesConversation()
    {
        var template = new ChatPromptTemplate();

        template.AddSystemMessage("You are a math tutor");
        template.AddUserMessage("What is 2+2?");
        template.AddAssistantMessage("2+2 equals 4");

        Assert.Equal(3, template.Messages.Count);
        Assert.Equal("system", template.Messages[0].Role);
        Assert.Equal("user", template.Messages[1].Role);
        Assert.Equal("assistant", template.Messages[2].Role);
    }

    [Fact(Timeout = 60000)]
    public async Task Format_ReturnsFormattedConversation()
    {
        var template = new ChatPromptTemplate();
        template.AddSystemMessage("You are helpful");
        template.AddUserMessage("Hello");

        var result = template.Format(new Dictionary<string, string>());

        Assert.Contains("System: You are helpful", result);
        Assert.Contains("User: Hello", result);
    }

    [Fact(Timeout = 60000)]
    public async Task ChainedCalls_Work()
    {
        var template = new ChatPromptTemplate()
            .AddSystemMessage("Be concise")
            .AddUserMessage("Hi")
            .AddAssistantMessage("Hello!");

        Assert.Equal(3, template.Messages.Count);
    }
}
