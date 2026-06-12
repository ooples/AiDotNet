using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Pipeline;
using AiDotNetTests.UnitTests.Agentic.Agents;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Pipeline
{
    /// <summary>
    /// Serializes the telemetry tests: the process-wide ActivitySource/Meter
    /// listeners would otherwise capture spans and metric points emitted by
    /// unrelated tests running in parallel, making Assert.Single flaky.
    /// </summary>
    [CollectionDefinition("AgenticTelemetry", DisableParallelization = true)]
    public sealed class AgenticTelemetryCollection
    {
    }

    [Collection("AgenticTelemetry")]
    public class TelemetryMiddlewareTests
    {
        [Fact(Timeout = 60000)]
        public async Task EmitsGenAiSpan_WithFinishReasonAndUsageTags()
        {
            await Task.Yield();

            var spans = new List<Activity>();
            using var listener = new ActivityListener
            {
                ShouldListenTo = source => source.Name == AgenticTelemetry.SourceName,
                Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllData,
                ActivityStopped = activity => spans.Add(activity),
            };
            ActivitySource.AddActivityListener(listener);

            var inner = ScriptedChatClient<double>.Sequence(
                ChatResponses.Text("hi", new ChatUsage(12, 5)));
            var client = new MiddlewareChatClient<double>(inner, new IChatMiddleware[] { new TelemetryChatMiddleware() });

            await client.GetResponseAsync(new[] { ChatMessage.User("hello") });

            var span = Assert.Single(spans);
            Assert.Equal("chat", span.GetTagItem("gen_ai.operation.name"));
            Assert.Equal(nameof(ChatFinishReason.Stop), span.GetTagItem("gen_ai.response.finish_reason"));
            Assert.Equal(12, span.GetTagItem("gen_ai.usage.input_tokens"));
            Assert.Equal(5, span.GetTagItem("gen_ai.usage.output_tokens"));
        }

        [Fact(Timeout = 60000)]
        public async Task EmitsOperationCountMetric()
        {
            await Task.Yield();

            var counts = new List<long>();
            using var meterListener = new MeterListener
            {
                InstrumentPublished = (instrument, l) =>
                {
                    if (instrument.Meter.Name == AgenticTelemetry.SourceName &&
                        instrument.Name == "gen_ai.client.operation.count")
                    {
                        l.EnableMeasurementEvents(instrument);
                    }
                },
            };
            meterListener.SetMeasurementEventCallback<long>((instrument, measurement, tags, state) => counts.Add(measurement));
            meterListener.Start();

            var inner = ScriptedChatClient<double>.Sequence(ChatResponses.Text("hi"));
            var client = new MiddlewareChatClient<double>(inner, new IChatMiddleware[] { new TelemetryChatMiddleware() });

            await client.GetResponseAsync(new[] { ChatMessage.User("hello") });
            meterListener.Dispose();

            Assert.Contains(1L, counts);
        }

        [Fact(Timeout = 60000)]
        public async Task NoListener_StillReturnsResponse()
        {
            await Task.Yield();

            // With no telemetry collector attached, the middleware is a transparent pass-through.
            var inner = ScriptedChatClient<double>.Sequence(ChatResponses.Text("passthrough"));
            var client = new MiddlewareChatClient<double>(inner, new IChatMiddleware[] { new TelemetryChatMiddleware() });

            var response = await client.GetResponseAsync(new[] { ChatMessage.User("hello") });

            Assert.Equal("passthrough", response.Text);
        }
    }
}
