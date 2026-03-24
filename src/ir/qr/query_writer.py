import requests


class QueryRewriter:
    def rewrite(self, query: str, n_queries=2):
        self.n_queries = n_queries


class OpenAIQueryRewriter(QueryRewriter):
    def __init__(self, client, n_queries=2):
        self.client = client
        self.n_queries = n_queries

    def rewrite(self, query):
        prompt = f"""
Rewrite the AWS question into {self.n_queries} search queries.
Question: {query}
Return only the queries.
"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        return response.choices[0].message.content


class LlamaQueryRewriter(QueryRewriter):
    def __init__(self, n_queries=2):
        self.n_queries = n_queries

    def rewrite(self, query):
        prompt = f"""
You are a search expert helping retrieve technical documentation.

Rewrite the following user query into {self.n_queries} alternative search queries
that would help retrieve relevant documentation.

Rules:
- Preserve meaning
- Use technical keywords
- Keep queries concise
- One query per line
- No numbering

User query:
{query}
"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False},
        )

        r = response.json()["response"]

        queries = [q.strip() for q in r.split("\n") if q.strip()]

        return list(queries)[1:]


if __name__ == "__main__":
    q = "concurrent lambda function executions limit"
    qw = LlamaQueryRewriter(2)
    print(qw.rewrite(q))
