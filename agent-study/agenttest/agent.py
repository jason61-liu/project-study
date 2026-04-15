"""
Author: liushiyi liushiyi2013@163.com
Date: 2026-03-02 23:40:27
LastEditors: liushiyi liushiyi2013@163.com
LastEditTime: 2026-03-27 23:30:57
FilePath: /pyproject/agent-study/agenttest/agent.py
Description:

Copyright (c) 2026 , All Rights Reserved.
"""

import uuid
import json
from typing import List, Optional, Callable, Dict
from pydantic import BaseModel, Field, ValidationError
import datetime


class WeatherRequest(BaseModel):
    location: str = Field(..., description="查询天气城市")
    date: Optional[str] = Field(
        default_factory=lambda: datetime.date.today(), description="查询日期"
    )


class TaskInput(BaseModel):
    task_id: str
    input: WeatherRequest


def weather_agent(input: WeatherRequest) -> Dict:
    return {
        "location": input.location,
        "date": str(input.date),
        "forecast": f"{input.location}天气晴，气温28",
    }


class AgentRegistry:
    def __init__(self):
        self._agents: Dict[str, Callable] = {}
        self._schemas: Dict[str, BaseModel] = {}

    def register(self, name: str, agent_fn: Callable, schema: BaseModel):
        self._agents[name] = agent_fn
        self._schemas[name] = schema

    def invoke(self, name: str, payload: Dict) -> Dict:
        if name not in self._agents:
            raise ValueError(f"Agent '{name}' not found")
        schema_cls = self._schemas[name]
        try:
            validated = schema_cls(**payload)
        except ValidationError as e:
            return {"error": str(e)}

        return self._agents[name](validated)


registry = AgentRegistry()
registry.register("weather_agent", weather_agent, WeatherRequest)
print("+++++++合法调用======")
valide_input = {"location": "北京"}
result = registry.invoke("weather_agent", valide_input)
print(json.dumps(result, ensure_ascii=False, indent=2))
