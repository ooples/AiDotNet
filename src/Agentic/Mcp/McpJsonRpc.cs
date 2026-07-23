using Newtonsoft.Json.Linq;

namespace AiDotNet.Agentic.Mcp;

/// <summary>
/// Helpers for building and parsing JSON-RPC 2.0 envelopes shared by the MCP transports.
/// </summary>
internal static class McpJsonRpc
{
    /// <summary>Builds a JSON-RPC 2.0 request envelope.</summary>
    public static JObject BuildRequest(int id, string method, JObject? parameters)
    {
        var envelope = new JObject
        {
            ["jsonrpc"] = "2.0",
            ["id"] = id,
            ["method"] = method,
        };
        if (parameters is not null)
        {
            envelope["params"] = parameters;
        }

        return envelope;
    }

    /// <summary>
    /// Parses a JSON-RPC 2.0 response body and returns its <c>result</c> object, throwing
    /// <see cref="McpException"/> on a JSON-RPC error or a malformed response.
    /// </summary>
    public static JObject ParseResult(string body)
    {
        if (body is null || body.Trim().Length == 0)
        {
            throw new McpException("Empty JSON-RPC response.");
        }

        JObject root;
        try
        {
            root = JObject.Parse(body);
        }
        catch (Newtonsoft.Json.JsonException ex)
        {
            throw new McpException("Malformed JSON-RPC response: " + ex.Message);
        }

        return ExtractResult(root);
    }

    /// <summary>
    /// Extracts the <c>result</c> object from an already-parsed JSON-RPC response, throwing
    /// <see cref="McpException"/> on a JSON-RPC error or a missing result.
    /// </summary>
    public static JObject ExtractResult(JObject root)
    {
        if (root["error"] is JObject error)
        {
            var message = (string?)error["message"] ?? "unknown error";
            var code = (int?)error["code"];
            throw new McpException(code is { } c ? $"MCP error {c}: {message}" : "MCP error: " + message);
        }

        if (root["result"] is JObject result)
        {
            return result;
        }

        throw new McpException("JSON-RPC response contained neither 'result' nor 'error'.");
    }
}
