import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Union

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.utils.reward_score.prime_math.grader import math_equal
from verl.utils.reward_score.gsm8k import extract_solution

@register("my_reward_manager")
class MyRewardManager:
    """
    针对Qwen2.5-VL-3B-Instruct数学数据集的定制奖励管理器。
    考虑因素：
    1. 答案正确性（准确率）- 主要目标
    2. 格式合规性（使用\boxed{}）
    3. 推理质量（思维链指标）
    4. 回答长度（递减回报以防止冗长）
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        
        # 权重和参数
        self.answer_weight = 2.0
        self.format_weight = 0.2
        self.length_weight = 0.2
        self.quality_weight = 0.3
        self.step_coverage_weight = 0.0
        
        self.max_tokens = 2048
        # CoT (思维链) 长度的目标参数
        # min_cot_length: 低于此长度不给予长度奖励（或奖励较低），防止过短的猜测
        # max_cot_length: 长度奖励的峰值点，超过此长度后奖励可能衰减
        self.min_cot_length = 512  # 原为 1500，对于简单问题 1500 太长了，调整为 500
        self.max_cot_length = self.max_tokens

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        # 尝试提取预计算的奖励（如果可用）
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        score_keys = ["format_score", "correctness_score", "reasoning_score", "length_score", "total_score"]
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]
            
            # 提取响应（解答）字符串
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            # 获取真实答案
            # 数据集结构是嵌套的：non_tensor_batch['reward_model']['ground_truth']
            # 我们安全地访问它
            reward_model_data = data_item.non_tensor_batch.get("reward_model", {})
            ground_truth = reward_model_data.get("ground_truth", "")
            
            # 如果ground_truth缺失或为空，则使用备用方法
            if not ground_truth:
                 # 尝试extra_info
                 extra_info = data_item.non_tensor_batch.get("extra_info", {})
                 ground_truth = extra_info.get("answer", "")

            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "unknown")

            # 计算自定义奖励
            score_result = self.compute_custom_reward(response_str, ground_truth)
            
            final_score = score_result["total_score"]
            
            # 将奖励分配给响应的最后一个token
            # 确保索引在边界内（valid_response_length - 1）
            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = final_score

            # 存储额外信息
            for key in score_keys:
                reward_extra_info[key].append(score_result.get(key, 0.0))

            # 日志记录/调试
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"\n[MyRewardManager 调试] 数据源: {data_source}")
                print(f"[响应]: {response_str[:200]}... (长度: {len(response_str)})")
                print(f"[真实答案]: {str(ground_truth)[:200]}...")
                for k, v in score_result.items():
                    print(f"  - {k}: {v}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def _extract_reward_from_rm_scores(self, data: DataProto, return_dict: bool):
        if "rm_scores" not in data.batch:
            return None
        if return_dict:
            return {"reward_tensor": data.batch["rm_scores"]}
        return data.batch["rm_scores"]

    def compute_custom_reward(self, response: str, ground_truth: Any) -> Dict[str, float]:
        """
        计算奖励组件。
        """
        components = {}
        base_reward = 0.0

        # --- 1. 答案提取和格式 ---
        extracted_answer = self.extract_answer(response)
        
        # 格式奖励：如果成功使用\boxed{}提取答案，给予小奖励
        if extracted_answer:
            base_reward += self.format_weight
            components["format_score"] = self.format_weight
        else:
            components["format_score"] = 0.0

        # --- 2. 答案正确性 ---
        # 确定目标答案字符串
        target_answer = None
        if isinstance(ground_truth, dict) and "answer" in ground_truth:
            target_answer = ground_truth["answer"]
        elif isinstance(ground_truth, (list, tuple)) and len(ground_truth) > 0:
             target_answer = str(ground_truth[0])
        else:
            target_answer = str(ground_truth)

        is_correct = False
        if extracted_answer and target_answer:
            # 使用verL工具中的math_equal
            try:
                is_correct = math_equal(extracted_answer, target_answer)
            except Exception as e:
                # 如果评分器失败，则退回到精确字符串匹配
                is_correct = (extracted_answer.strip() == target_answer.strip())
        
        if is_correct:
            base_reward += self.answer_weight
            components["correctness_score"] = self.answer_weight
        else:
            # 错误答案的惩罚
            base_reward -= 1.0
            components["correctness_score"] = -1.0
            
            # 如果不正确，我们返回当前分数（格式 + 正确性惩罚）并提前结束
            # 我们跳过推理/长度奖励，以避免奖励幻觉的长链
            components["total_score"] = base_reward
            return components

        # --- 3. 推理质量 ---
        # 仅在答案正确时评估（或者如果我们决定宽松处理）
        # 这里因为答案正确，我们继续
        
        gt_steps = None
        if isinstance(ground_truth, dict) and "steps" in ground_truth:
            gt_steps = ground_truth["steps"]
        
        reasoning_score = self.evaluate_reasoning_quality(response, gt_steps)
        base_reward += reasoning_score * self.quality_weight
        components["reasoning_score"] = reasoning_score

        # --- 4. 长度奖励（递减回报） ---
        token_count = self.count_tokens(response)
        if token_count <= self.max_tokens:
            length_score = self.length_reward_curve(token_count)
            base_reward += length_score * self.length_weight
            components["length_score"] = length_score
        else:
            base_reward -= 0.5
            components["length_score"] = -0.5

        base_reward = max(min(base_reward, 2.5), -1.5)
        components["total_score"] = base_reward
        return components

    def extract_answer(self, response: str) -> str:
        """
        从响应字符串中提取答案。
        处理\boxed{...}的嵌套花括号，正则表达式无法很好处理。
        """
        if not response:
            return ""
            
        # 1. 尝试健壮的嵌套花括号提取\boxed{}
        results = []
        idx = response.find("\\boxed{")
        while idx != -1:
            brace_count = 1
            start_content = idx + 7 # len("\\boxed{") 是 7
            current = start_content
            content = ""
            while current < len(response) and brace_count > 0:
                if response[current] == '{':
                    brace_count += 1
                elif response[current] == '}':
                    brace_count -= 1
                
                if brace_count > 0:
                    content += response[current]
                current += 1
                
            if brace_count == 0:
                results.append(content)
            
            # 搜索下一个出现位置
            idx = response.find("\\boxed{", start_content)
            
        if results:
            return results[-1] # 返回最后一个boxed内容
        
        # 2. 备用：尝试verl的灵活提取器（GSM8K风格）
        try:
            extracted = extract_solution(response, method="flexible")
            if extracted:
                return extracted
        except:
            pass
            
        return ""

    def evaluate_reasoning_quality(self, response: str, gt_steps: List[str] = None) -> float:
        """
        基于指标和结构评估推理质量。
        """
        score = 0.0
        
        # 1. 逻辑指示词
        reasoning_indicators = ["because", "so", "therefore", "due to", "conclude", "implies", "since", "thus", "step"]
        indicator_count = sum(1 for indicator in reasoning_indicators if indicator in response.lower())
        score += min(indicator_count * 0.05, 0.4) # 指标最大0.4分
        
        # 2. 数学术语
        math_terms = ["set", "let", "substitute", "simplify", "prove", "calculate", "derive", "assume", "equation"]
        term_count = sum(1 for term in math_terms if term in response.lower())
        score += min(term_count * 0.05, 0.4) # 术语最大0.4分

        # 3. 结构检查（例如段落或换行）
        if response.count("\n") > 5:
            score += 0.2
            
        return min(score, 1.0)

    def count_tokens(self, text: str) -> int:
        """
        使用分词器或简单分割（如果分词器失败）近似token数量。
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text.split())

    def length_reward_curve(self, token_count: int) -> float:
        """
        基于长度偏好返回分数：
        - < min_cot_length: 0 分 (鼓励更长的 CoT，但不要太短)
        - min_cot_length - max_cot_length: 线性增加的分数
        - > max_cot_length: 分数递减 (防止过度冗长)
        """
        if token_count < self.min_cot_length:
            return 0.0
        
        if token_count <= self.max_cot_length:
            # 在 min 和 max 之间线性增加，从 0.0 到 1.0
            return (token_count - self.min_cot_length) / (self.max_cot_length - self.min_cot_length)
        
        else:
            # > max_cot_length: 衰减
            # 超过峰值后，每 1000 tokens 减少 0.1 分
            
            excess = token_count - self.max_cot_length
            decay = excess / 1000.0
            score = 1.0 - decay
            return max(score, 0.0) # 下限为 0
