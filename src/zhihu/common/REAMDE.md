# README: usage of api

- 可以发起单独的一次和LLM的对话

```python
# 和LLM的一次单轮对话
rsp = get_completion("讲一个10个字以内的笑话")

```

- 可以发起和LLM的多轮对话，同时可以设定message role为system的prompt
```python
# 可以预先设定system prompt，设定之后的调用get_completion(), 就自动含有该system prompt
set_system_prompt("你是一个说唱歌手.")
get_completion("请创作一段说唱音乐歌词, 不超过20个字。", clear_session=False)
session_msg = get_session_message()
assert session_msg[0]['role'] == 'system'
```
    + clear_session 参数：默认情况下, get_completion执行完一次会话，会将_session_message清空, 如果希望进行携带着之前的会话信息，则需要显式的指定clear_sesson False不清空_session_message
    + 多轮对话sample如下

```python
rsp = get_completion("请告诉我去沙河水库怎么走?", "你是一个北京市的地理通", clear_session=False)
add_session_message("assistant", rsp.choice[0].message.content)
rsp = get_completion("我改主意不去沙河了，我去天坛南门去喝豆汁，请重新告诉我怎么走")
logger.info("rsp=%$", rsp.choices[0].message.content)
```

- 在调用get_completion()之前, _session_message可能是有内容的，如果你是希望启动单独一次和LLM的会话，应该调用reset_session_message()
<div>
    ![error](../images/need_reset_msg.png)
</div>
    + 上图是没有清除session_message时, 设定system_prompt报错