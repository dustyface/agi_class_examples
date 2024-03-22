# README: function calling

## 启动Function Calling的要点

1. 必须使用支持functio calling的model: OpenAI的[Function Calling](https://platform.openai.com/docs/guides/function-calling),说明了可以通过检测message的格式，支持function calling的model, 最近的比较新的model是:
    + `gpt-3.5-turbo-0125`
    + `gpt-4-turbo-preview`
    + 注意，声明的较早一些的model, e.g. gpt-3.5-turbo-1106, 实测中它已经不支持function calling了
2. message的格式中必须要包括tools字段，具体它的定义，参考代码sum.py等
3. 在第一轮和OpenAI LLM交互之后, 必须把返回的response的message(role是assistant)加入到message列表中。然后再填入role为tool的message, 再启动第2轮和LLM的对话; i.e. 一个function calling的message的role的顺序典型的是: system->user->assistant->tool