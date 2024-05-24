from typing import Any, List
from urllib.parse import quote
#用来异步http请求
import aiohttp
from aiohttp import ClientResponseError
from fastapi.responses import StreamingResponse as FastAPIStreamingResponse
from loguru import logger

from reworkd_platform.settings import settings
from reworkd_platform.web.api.agent.stream_mock import stream_string
from reworkd_platform.web.api.agent.tools.reason import Reason
from reworkd_platform.web.api.agent.tools.tool import Tool
from reworkd_platform.web.api.agent.tools.utils import (
    CitedSnippet,
    summarize_with_sources,
)

# Search google via serper.dev. Adapted from LangChain
# https://github.com/hwchase17/langchain/blob/master/langchain/utilities

#使用Serper API执行Google搜索 并返回搜索结果
async def _google_serper_search_results(
    #搜索的查询字符串，  搜索的类型 默认值为search
    search_term: str, search_type: str = "search"
) -> dict[str, Any]:   #函数签名
    headers = {
        "X-API-KEY": settings.serp_api_key or "",
        "Content-Type": "application/json",
    }
    params = {
        "q": search_term,
    }
    #使用 aiohttp 创建一个异步会话，并发送 POST 请求
    async with aiohttp.ClientSession() as session:
        #发送 POST 请求到 Serper API，传递请求头和查询参数
        async with session.post(
            f"https://google.serper.dev/{search_type}", headers=headers, params=params
        ) as response:
            response.raise_for_status()
            #解析响应 JSON 数据并返回结果
            search_results = await response.json()
            return search_results


class Search(Tool):
    #在谷歌上搜索有关公共信息""新闻和人物"
    description = (
        "Search Google for short up to date searches for simple questions about public information "
        "news and people.\n"
    )
    #在谷歌上搜索有关当前事件的信息
    public_description = "Search google for information about current events."
    #要搜索的查询参数。此值总是填充的，不能为空字符串。
    arg_description = "The query argument to search for. This value is always populated and cannot be an empty string."
    image_url = "/tools/google.png"

    @staticmethod
    # 可用性检查（检查serp_api_key是否已配置）
    def available() -> bool:
        return settings.serp_api_key is not None and settings.serp_api_key != ""

    async def call(
        self, goal: str, task: str, input_str: str, *args: Any, **kwargs: Any
    ) -> FastAPIStreamingResponse:
        try:
            #调用 _call 方法执行搜索
            return await self._call(goal, task, input_str, *args, **kwargs)
        #如果出现 ClientResponseError 异常
        except ClientResponseError:
            #记录异常
            logger.exception("Error calling Serper API, falling back to reasoning")
            #如果搜索失败，回退到 Reason 类进行推理
            return await Reason(self.model, self.language).call(
                goal, task, input_str, *args, **kwargs
            )

    async def _call(
        self, goal: str, task: str, input_str: str, *args: Any, **kwargs: Any
    ) -> FastAPIStreamingResponse:
        #使用 Google 搜索工具
        results = await _google_serper_search_results(
            input_str,
        )
        # 返回5个
        k = 5  # Number of results to return
        snippets: List[CitedSnippet] = []
        #检查搜索结果中是否包含 answerBox
        if results.get("answerBox"):
            answer_values = []
            answer_box = results.get("answerBox", {})
            #提取答案
            if answer_box.get("answer"):
                answer_values.append(answer_box.get("answer"))
            #提取片段
            elif answer_box.get("snippet"):
                answer_values.append(answer_box.get("snippet").replace("\n", " "))
            #提取高亮片段
            elif answer_box.get("snippetHighlighted"):
                answer_values.append(", ".join(answer_box.get("snippetHighlighted")))

            if len(answer_values) > 0:
                #添加到 snippets 列表中
                snippets.append(
                    CitedSnippet(
                        len(snippets) + 1,
                        "\n".join(answer_values),
                        f"https://www.google.com/search?q={quote(input_str)}",
                    )
                )
        #遍历 organic 结果
        for i, result in enumerate(results["organic"][:k]):
            texts = []
            link = ""
            #提取片段
            if "snippet" in result:
                texts.append(result["snippet"])
            #提取链接
            if "link" in result:
                link = result["link"]
            for attribute, value in result.get("attributes", {}).items():
                texts.append(f"{attribute}: {value}.")
            #添加到 snippets 列表
            snippets.append(CitedSnippet(len(snippets) + 1, "\n".join(texts), link))
        #没有找到搜索结果
        if len(snippets) == 0:
            return stream_string("No good Google Search Result was found", True)
        #根据搜索结果生成总结并返回流式响应
        return summarize_with_sources(self.model, self.language, goal, task, snippets)
