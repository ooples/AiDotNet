using System.Net;
using System.Net.Http;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.ProgramSynthesis.Results;
using AiDotNet.ProgramSynthesis.Serving;
using Newtonsoft.Json;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ProgramSynthesis;

public sealed class ProgramSynthesisServingClientTests
{
    [Fact]
    public void Ctor_BaseAddressMissing_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new ProgramSynthesisServingClient(new ProgramSynthesisServingClientOptions()));
    }

    [Fact]
    public async Task ExecuteProgramAsync_SendsAuthHeadersAndDeserializesResponse()
    {
        var handler = new RecordingHandler(_ =>
        {
            var json = JsonConvert.SerializeObject(new ProgramExecuteResponse
            {
                Success = true,
                Language = ProgramLanguage.CSharp,
                ExitCode = 0
            });

            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json")
            };
        });

        var httpClient = new HttpClient(handler) { BaseAddress = new Uri("http://localhost:1234/") };
        var options = new ProgramSynthesisServingClientOptions
        {
            BaseAddress = new Uri("http://localhost:1234/"),
            HttpClient = httpClient,
            ApiKeyHeaderName = "X-Api-Key",
            ApiKey = "abc",
            BearerToken = "token"
        };

        var client = new ProgramSynthesisServingClient(options);
        var response = await client.ExecuteProgramAsync(new ProgramExecuteRequest
        {
            Language = ProgramLanguage.CSharp,
            SourceCode = "Console.WriteLine(1);"
        }, CancellationToken.None);

        Assert.True(response.Success);
        Assert.Equal(ProgramLanguage.CSharp, response.Language);
        Assert.NotNull(handler.LastRequest);
        Assert.True(handler.LastRequest!.Headers.TryGetValues("X-Api-Key", out var values));
        Assert.Contains("abc", values);
        Assert.NotNull(handler.LastRequest.Headers.Authorization);
        Assert.Equal("Bearer", handler.LastRequest.Headers.Authorization!.Scheme);
        Assert.Equal("token", handler.LastRequest.Headers.Authorization!.Parameter);
    }

    [Fact]
    public async Task ExecuteProgramAsync_UnparseableBody_Throws()
    {
        var handler = new RecordingHandler(_ =>
            new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent("not-json", System.Text.Encoding.UTF8, "application/json")
            });

        var client = new ProgramSynthesisServingClient(new ProgramSynthesisServingClientOptions
        {
            BaseAddress = new Uri("http://localhost:1234/"),
            HttpClient = new HttpClient(handler) { BaseAddress = new Uri("http://localhost:1234/") }
        });

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            client.ExecuteProgramAsync(new ProgramExecuteRequest { Language = ProgramLanguage.Python, SourceCode = "print(1)" }, CancellationToken.None));
    }

    [Fact]
    public async Task ExecuteProgramAsync_NonSuccessAndUnparseableBody_ThrowsHttpRequestException()
    {
        var handler = new RecordingHandler(_ =>
            new HttpResponseMessage(HttpStatusCode.InternalServerError)
            {
                Content = new StringContent("not-json", System.Text.Encoding.UTF8, "application/json")
            });

        var client = new ProgramSynthesisServingClient(new ProgramSynthesisServingClientOptions
        {
            BaseAddress = new Uri("http://localhost:1234/"),
            HttpClient = new HttpClient(handler) { BaseAddress = new Uri("http://localhost:1234/") }
        });

        await Assert.ThrowsAsync<HttpRequestException>(() =>
            client.ExecuteProgramAsync(new ProgramExecuteRequest { Language = ProgramLanguage.Python, SourceCode = "print(1)" }, CancellationToken.None));
    }

    [Fact]
    public async Task ExecuteCodeTaskAsync_Summarization_UsesTaskRoute()
    {
        var handler = new RecordingHandler(request =>
        {
            Assert.Equal("/api/program-synthesis/tasks/summarization", request.RequestUri!.AbsolutePath);

            var json = JsonConvert.SerializeObject(new CodeSummarizationResult
            {
                Success = true,
                Language = ProgramLanguage.CSharp,
                Summary = "ok"
            });

            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json")
            };
        });

        var client = new ProgramSynthesisServingClient(new ProgramSynthesisServingClientOptions
        {
            BaseAddress = new Uri("http://localhost:1234/"),
            HttpClient = new HttpClient(handler) { BaseAddress = new Uri("http://localhost:1234/") }
        });

        var result = await client.ExecuteCodeTaskAsync(
            new CodeSummarizationRequest { Language = ProgramLanguage.CSharp, Code = "class C {}" },
            CancellationToken.None);

        var typed = Assert.IsType<CodeSummarizationResult>(result);
        Assert.True(typed.Success);
        Assert.Equal("ok", typed.Summary);
    }

    [Fact]
    public async Task ExecuteCodeTaskAsync_UnsupportedRequestType_Throws()
    {
        var handler = new RecordingHandler(_ => new HttpResponseMessage(HttpStatusCode.OK));

        var client = new ProgramSynthesisServingClient(new ProgramSynthesisServingClientOptions
        {
            BaseAddress = new Uri("http://localhost:1234/"),
            HttpClient = new HttpClient(handler) { BaseAddress = new Uri("http://localhost:1234/") }
        });

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            client.ExecuteCodeTaskAsync(new UnsupportedCodeTaskRequest(), CancellationToken.None));
    }

    private sealed class RecordingHandler : HttpMessageHandler
    {
        private readonly Func<HttpRequestMessage, HttpResponseMessage> _handler;

        public RecordingHandler(Func<HttpRequestMessage, HttpResponseMessage> handler)
        {
            _handler = handler;
        }

        public HttpRequestMessage? LastRequest { get; private set; }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            LastRequest = request;
            return Task.FromResult(_handler(request));
        }
    }

    private sealed class UnsupportedCodeTaskRequest : CodeTaskRequestBase
    {
        public override CodeTask Task => CodeTask.Generation;
    }
}
