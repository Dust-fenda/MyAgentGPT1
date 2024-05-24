from typing import TypeVar

from fastapi import Body, Depends
#用 SQLAlchemy 进行数据库操作
from sqlalchemy.ext.asyncio import AsyncSession

from reworkd_platform.db.crud.agent import AgentCRUD
from reworkd_platform.db.dependencies import get_db_session
from reworkd_platform.schemas.agent import (
    AgentChat,
    AgentRun,
    AgentRunCreate,
    AgentSummarize,
    AgentTaskAnalyze,
    AgentTaskCreate,
    AgentTaskExecute,
    Loop_Step,
)
from reworkd_platform.schemas.user import UserBase
from reworkd_platform.web.api.dependencies import get_current_user
#泛型类型 T, T 可以是 AgentTaskAnalyze, AgentTaskExecute ..AgentChat等
T = TypeVar(
    "T", AgentTaskAnalyze, AgentTaskExecute, AgentTaskCreate, AgentSummarize, AgentChat
)


def agent_crud(
    user: UserBase = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> AgentCRUD:
    return AgentCRUD(session, user)


async def agent_start_validator(
    body: AgentRunCreate = Body(
        example={
            "goal": "Create business plan for a bagel company",
            "modelSettings": {
                "customModelName": "gpt-3.5-turbo",
            },
        },
    ),
    crud: AgentCRUD = Depends(agent_crud),
) -> AgentRun:  #函数签名
    id_ = (await crud.create_run(body.goal)).id
    return AgentRun(**body.dict(), run_id=str(id_))

#通用的验证函数 validate ， crud: 一个 AgentCRUD 实例 ，type_: 一个 Loop_Step 类型的值，用于指定任务的类型
async def validate(body: T, crud: AgentCRUD, type_: Loop_Step) -> T:
    body.run_id = (await crud.create_task(body.run_id, type_)).id
    return body


async def agent_analyze_validator(
    body: AgentTaskAnalyze = Body(),
    crud: AgentCRUD = Depends(agent_crud),
) -> AgentTaskAnalyze:
    return await validate(body, crud, "analyze")


#验证并处理一个执行任务(AgentTaskExecute)
async def agent_execute_validator(
    body: AgentTaskExecute = Body(
        example={
            "goal": "Perform tasks accurately",
            "task": "Write code to make a platformer",
            "analysis": {
                "reasoning": "I like to write code.",
                "action": "code",
                "arg": "",
            },
        },
    ),
    #依赖于 agent_crud 来获取 AgentCRUD 实例
    crud: AgentCRUD = Depends(agent_crud),
) -> AgentTaskExecute:
    #使用 validate 函数来完成实际的验证和处理工作
    return await validate(body, crud, "execute")


async def agent_create_validator(
    body: AgentTaskCreate = Body(),
    crud: AgentCRUD = Depends(agent_crud),
) -> AgentTaskCreate:
    return await validate(body, crud, "create")


async def agent_summarize_validator(
    body: AgentSummarize = Body(),
    crud: AgentCRUD = Depends(agent_crud),
) -> AgentSummarize:
    return await validate(body, crud, "summarize")


async def agent_chat_validator(
    body: AgentChat = Body(),
    crud: AgentCRUD = Depends(agent_crud),
) -> AgentChat:
    return await validate(body, crud, "chat")
