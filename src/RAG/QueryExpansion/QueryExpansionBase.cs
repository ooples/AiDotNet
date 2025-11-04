namespace AiDotNet.RAG.QueryExpansion
{
    public abstract class QueryExpansionBase<T>
    {
        public List<string> Expand(string query, Dictionary<string, object>? options = null)
        {
            if (string.IsNullOrWhiteSpace(query))
            {
                throw new ArgumentException("Query cannot be null or empty", nameof(query));
            }

            return ExpandCore(query, options);
        }

        protected abstract List<string> ExpandCore(string query, Dictionary<string, object>? options = null);
    }
}
