""" A fast server for testing """
#!/usr/bin/env python
import logging
from fastapi import FastAPI
from langserve import add_routes
from langchain_samples.tutorial.translator import simple_translate_with_chain

logger = logging.getLogger(__name__)

# App definition
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

chain = simple_translate_with_chain()

# langserve 把chain封装成一个controller
add_routes(
    app,
    chain,
    path="/chain",
)

# run:
# python -m zhihu.LangChain.tutorial.fast_server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
