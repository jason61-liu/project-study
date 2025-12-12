import os
import re
import json
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

import requests
from tavily import TavilyClient
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 配置日志
log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置常量
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "5"))
WEATHER_API_URL = "https://wttr.in/{city}?format=j1"
DEFAULT_SEARCH_DEPTH = "basic"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# 系统提示词
AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点

# 行动格式:你的回答必须严格遵循以下格式。首先是你的思考过程，然后是你要执行的具体行动。
Thought: [这里是你的思考过程和下一步计划]
Action: [这里是你要调用的工具，格式为 function_name(arg_name="arg_value")]

# 任务完成:当你收集到足够的信息，能够回答用户的最终问题时，你必须使用 `finish(answer="...")` 来输出最终答案。

请开始吧！
"""


class TravelTools:
    """旅行相关工具集合"""

    def __init__(self):
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not self.tavily_api_key:
            logger.warning("TAVILY_API_KEY environment variable not set")

    def get_weather(self, city: str) -> str:
        """查询指定城市的实时天气"""
        if not city or not city.strip():
            return "Error: City name cannot be empty"

        url = WEATHER_API_URL.format(city=city.strip())
        try:
            logger.info(f"Querying weather information for {city}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            current_condition = data["current_condition"][0]
            weather_desc = current_condition["weatherDesc"][0]["value"]
            temp_c = current_condition["temp_C"]

            result = f"{city} 当前天气：{weather_desc}，气温：{temp_c}摄氏度"
            logger.info(f"天气查询成功：{result}")
            return result

        except requests.exceptions.RequestException as e:
            error_msg = f"Network issue encountered while querying weather: {e}"
            logger.error(error_msg)
            return error_msg
        except (KeyError, IndexError) as e:
            error_msg = f"Failed to parse weather data, possibly invalid city name: {e}"
            logger.error(error_msg)
            return error_msg

    def get_attraction(self, city: str, weather: str) -> str:
        """根据城市和天气搜索推荐的旅游景点"""
        if not city or not city.strip():
            return "Error: City name cannot be empty"
        if not weather or not weather.strip():
            return "Error: Weather information cannot be empty"

        if not self.tavily_api_key:
            return "Error: Tavily API key not configured"

        try:
            logger.info(f"Searching for attraction recommendations in {city} during {weather} weather")
            tavily = TavilyClient(api_key=self.tavily_api_key)
            query = f"{city} 在 {weather}天气下最适合去的旅游景点推荐及理由"

            response = tavily.search(
                query=query,
                search_depth=DEFAULT_SEARCH_DEPTH,
                include_answer=True
            )

            if response.get("answer"):
                result = response["answer"]
                logger.info("Attraction search successful, using direct answer")
                return result

            formatted_results = []
            for result in response.get("results", []):
                formatted_results.append(f"- {result['title']}: {result['content']}")

            if not formatted_results:
                result = "Sorry, no relevant tourist attraction information found"
                logger.warning(result)
                return result

            result = "Based on the search, we found the following information for you:\n" + "\n".join(formatted_results)
            logger.info("Attraction search successful, using search results")
            return result

        except Exception as e:
            error_msg = f"Error occurred while performing attraction search: {e}"
            logger.error(error_msg)
            return error_msg


class LLMClient:
    """大语言模型客户端"""

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """生成模型响应"""
        logger.info("Calling large language model...")
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            logger.info("Large language model response successful")
            return answer

        except Exception as e:
            error_msg = f"Error occurred while calling LLM API: {e}"
            logger.error(error_msg)
            return "Error: Failed to call language model service"


class ActionParser:
    """Action解析器"""

    @staticmethod
    def parse_action(llm_output: str) -> Optional[Dict[str, Any]]:
        """解析模型输出的Action"""
        action_match = re.search(r"Action:(.*)", llm_output, re.DOTALL)
        if not action_match:
            logger.error("Action not found in model output")
            return None

        action_str = action_match.group(1).strip()

        # 检查是否是完成动作
        if action_str.startswith("finish"):
            finish_match = re.search(r'finish\(answer="(.*)"\)', action_str)
            if finish_match:
                return {"type": "finish", "answer": finish_match.group(1)}
            else:
                logger.error("Unable to parse finish action parameters")
                return None

        # 解析工具调用
        tool_match = re.search(r"(\w+)\((.*)\)", action_str)
        if not tool_match:
            logger.error(f"Unable to parse tool call: {action_str}")
            return None

        tool_name = tool_match.group(1)
        args_str = tool_match.group(2)
        kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

        return {
            "type": "tool_call",
            "tool_name": tool_name,
            "args": kwargs
        }


class TravelAgent:
    """智能旅行助手主程序"""

    def __init__(self):
        self.tools = TravelTools()
        self.available_tools = {
            "get_weather": self.tools.get_weather,
            "get_attraction": self.tools.get_attraction
        }

        # 从环境变量获取配置
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")

        self.llm_client = LLMClient(model, api_key, base_url)
        self.action_parser = ActionParser()

    def run(self, user_prompt: str, max_rounds: int = MAX_ROUNDS) -> str:
        """运行旅行助手"""
        prompt_history = [f"用户请求：{user_prompt}"]
        logger.info(f"Starting to process user request: {user_prompt}")

        print(f"用户输入：{user_prompt}\n" + "=" * 40)

        for round_num in range(max_rounds):
            print(f"---第{round_num + 1}轮对话---\n")

            full_prompt = "\n".join(prompt_history)
            llm_output = self.llm_client.generate(full_prompt, AGENT_SYSTEM_PROMPT)

            print(f"模型输出：\n{llm_output}\n")
            prompt_history.append(llm_output)

            # 解析Action
            action = self.action_parser.parse_action(llm_output)
            if not action:
                print("Parse error: Unable to parse Action from model output.")
                break

            # 处理完成动作
            if action["type"] == "finish":
                final_answer = action["answer"]
                print(f"任务完成，最终答案: {final_answer}")
                logger.info("Task completed successfully")
                return final_answer

            # 处理工具调用
            if action["type"] == "tool_call":
                tool_name = action["tool_name"]
                args = action["args"]

                if tool_name in self.available_tools:
                    observation = self.available_tools[tool_name](**args)
                else:
                    observation = f"Error: Undefined tool '{tool_name}'"
                    logger.warning(f"Attempted to call undefined tool: {tool_name}")

                observation_str = f"Observation: {observation}"
                print(f"{observation_str}\n" + "=" * 40)
                prompt_history.append(observation_str)

        logger.warning("Maximum conversation rounds reached, task not completed")
        return "Sorry, unable to complete your request due to reaching maximum conversation rounds."


if __name__ == "__main__":
    try:
        # 创建并运行旅行助手
        agent = TravelAgent()
        user_prompt = "你好，请帮我查询一下今天深圳天气，然后根据天气推荐一个合适的旅游景点"
        result = agent.run(user_prompt)

    except Exception as e:
        logger.error(f"Program runtime error: {e}")
        print(f"Program runtime error: {e}")