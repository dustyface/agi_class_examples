""" Test LangFuse """
from langfuse.decorators import observe
from langfuse.openai import openai    # 需要使用langfuse对OpenAI LLM的集成包


@observe()
def run():
    """ hello to langfuse """
    return openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "对我说hello world"}
        ]).choices[0].message.content
