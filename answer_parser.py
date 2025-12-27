"""
答案解析器：从模型输出中提取答案和推理原因
"""

import re
from typing import Dict, Optional, Tuple

class AnswerParser:
    """答案解析器"""
    
    @staticmethod
    def parse_answer_with_reason(output: str) -> Tuple[str, str]:
        """
        从模型输出中解析答案和推理原因
        
        Args:
            output: 模型原始输出
            
        Returns:
            (answer, reason) 元组，如果解析失败则返回 (output, "")
        """
        output = output.strip()
        
        # 尝试解析格式：Answer: [letters]\nReason: [text]
        # 答案部分应该只匹配到换行符或 "Reason:" 之前
        answer_pattern = r'Answer:\s*([A-Z,;\s]+?)(?:\n|Reason:|$)'
        reason_pattern = r'Reason:\s*(.+?)(?:\n\n|\Z)'
        
        answer_match = re.search(answer_pattern, output, re.IGNORECASE | re.MULTILINE)
        reason_match = re.search(reason_pattern, output, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        
        if answer_match:
            answer = answer_match.group(1).strip()
            # 清理答案，只保留字母和逗号/分号
            answer = re.sub(r'[^A-Z,;]', '', answer.upper())
            reason = reason_match.group(1).strip() if reason_match else ""
            return answer, reason
        
        # 如果没有找到Answer:格式，尝试直接提取选项字母
        # 提取所有选项字母（A-Z）
        letters = re.findall(r'\b([A-Z])\.?\b', output)
        if letters:
            # 去重并保持顺序
            seen = set()
            unique_letters = []
            for letter in letters:
                if letter not in seen:
                    seen.add(letter)
                    unique_letters.append(letter)
            
            answer = ",".join(unique_letters)
            
            # 尝试提取reason（在答案之后的内容）
            # 查找"Reason:"关键词
            reason_match = re.search(r'Reason:\s*(.+?)(?:\n\n|\Z)', output, re.IGNORECASE | re.DOTALL)
            if reason_match:
                reason = reason_match.group(1).strip()
            else:
                # 如果没有找到Reason:，尝试提取答案之后的所有文本作为reason
                # 移除答案部分
                answer_removed = re.sub(r'\b([A-Z])\.?\s*[^\s]*(?:\s+[A-Z]\.?\s*[^\s]*)*', '', output)
                answer_removed = re.sub(r'Answer:.*', '', answer_removed, flags=re.IGNORECASE)
                reason = answer_removed.strip()
            
            return answer, reason
        
        # 如果都解析失败，返回原始输出作为answer
        return output, ""
    
    @staticmethod
    def extract_answer_letters(text: str) -> str:
        """
        从文本中提取答案字母（用于评估）
        
        Args:
            text: 答案文本，可能是 "A,B" 或 "A. Normal, B. Reduced" 等格式
            
        Returns:
            提取的字母字符串，如 "A,B"
        """
        # 提取所有选项字母
        letters = re.findall(r'\b([A-Z])\.?\b', text)
        if letters:
            # 去重并保持顺序
            seen = set()
            unique_letters = []
            for letter in letters:
                if letter not in seen:
                    seen.add(letter)
                    unique_letters.append(letter)
            return ",".join(unique_letters)
        
        # 如果没有找到字母，返回原始文本（去除空格）
        return text.replace(" ", "").replace(",", ",")

