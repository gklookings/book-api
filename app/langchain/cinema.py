from openai import OpenAI


def query_cinema(query: str):
    try:
        client = OpenAI()
        input = f"Based on the provided query, please provide a concise answer. If the query is not related to cinema, respond with 'The query is not related to cinema'. Query: {query}"
        response = client.responses.create(
            model="gpt-4.1-mini",
            tools=[
                {
                    "type": "web_search_preview",
                    "search_context_size": "medium",
                }
            ],
            input=input,
        )
        return {
            "answer": response.output_text,
        }, 200
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}
