from typing import Any

from fastapi.responses import StreamingResponse as FastAPIStreamingResponse
from lanarky.responses import StreamingResponse
from langchain import LLMChain

from reworkd_platform.web.api.agent.tools.tool import Tool

#用的是大模型的推理功能
class Reason(Tool):
    description = (
        #根据现有的信息、理解 进行 任务推理
        "Reason about task via existing information or understanding. "
        #从选项中做出决策
        "Make decisions / selections from options."
    )

    async def call(
        self, goal: str, task: str, input_str: str, *args: Any, **kwargs: Any
    ) -> FastAPIStreamingResponse:
        #导入 execute_task_prompt
        from reworkd_platform.web.api.agent.prompts import execute_task_prompt
        #也是用LangChain框架 创建 LLMChain 实例，传入语言模型和提示模板
        chain = LLMChain(llm=self.model, prompt=execute_task_prompt)
        #返回一个流式响应 调用 StreamingResponse.from_chain 方法
        return StreamingResponse.from_chain(
            chain,
            {"goal": goal, "language": self.language, "task": task},
            media_type="text/event-stream",
        )
