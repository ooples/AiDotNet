using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System.Collections.Generic;

namespace AiDotNet.RAG.ContextCompression
{
    public abstract class ContextCompressorBase<T>
        where T : struct, IComparable, IConvertible, IFormattable
    {
        public List<Document<T>> Compress(
            List<Document<T>> documents,
            string query,
            Dictionary<string, object>? options = null)
        {
            if (documents == null || documents.Count == 0)
            {
                return new List<Document<T>>();
            }

            return CompressCore(documents, query, options);
        }

        protected abstract List<Document<T>> CompressCore(
            List<Document<T>> documents,
            string query,
            Dictionary<string, object>? options = null);
    }
}
