'''
Author: jason61-liu jason61-liu@users.noreply.github.com
Date: 2026-06-10 14:37:54
LastEditors: jason61-liu jason61-liu@users.noreply.github.com
LastEditTime: 2026-06-10 14:37:58
FilePath: /test/test.py
Description: 

Copyright (c) 2026 , All Rights Reserved. 
'''

import json

def main(http_response: str) -> dict:
      # 如果 http_response 是字符串，先解析
      if isinstance(http_response, str):
          data = json.loads(http_response)
      else:
          data = http_response

      results = data.get("results", [])

      # 提取所有 memory 内容
      memories = [item.get("memory", "") for item in results]

      # 按 score 排序（通常已经是降序）
      sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)

      return {
          "memories": memories,
          "top_memory": sorted_results[0].get("memory", "") if sorted_results else "",
          "top_score": sorted_results[0].get("score", 0) if sorted_results else 0,
          "memory_count": len(results),
          "all_ids": [item.get("id", "") for item in results]
      }
