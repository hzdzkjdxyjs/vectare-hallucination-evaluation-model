import pandas as pd
import json
from transformers import AutoModelForSequenceClassification
import torch
import time
from statistics import mean
import os
from datetime import datetime
from tqdm.asyncio import tqdm_asyncio
import aiohttp
import asyncio
import aiofiles

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))   # 获得项目地址
#---------------------------设置参数----------------------------------------
# 初始化后端
BACKEND_URL = "http://0.0.0.0:9997"
BACKEND = 0                                                 # 选择什么后端是vllm则填0，xinference填1，该选项不影响结果，仅作为记录用
MODELUID = "deepseek/deepseek-r1"                           # 模型uid 必须改
MODELNAME = "deepseek/deepseek-r1"                          # 自定义的name，无所谓能区分就行
THINKING = True                                             # 是否开启思考模式
MAX_TEST_SAMPLES = 5                                        # 可调整此值控制测试数据量,一般不超过1006
CONCURRENCY_LIMIT = 5                                       # 并发数
USE_SYSTEM_PROMPT = False                                   # 是否使用系统提示词
GENERATE_CONFIG =  {
    "max_tokens": 1024,                                     # 最大生成的token数量
    "top_k": 50,                                            # top-k采样，限制候选词数量
    "temperature": 0,                                       # 温度，控制生成随机性
    # "top_p": 0.9,                                         # 核采样，限制累积概率
    # "presence_penalty": 0.6,                              # 惩罚已生成的token
    "frequency_penalty": 0.5,                               # 惩罚重复的token
    # "stop": ["</s>", "\n"],                               # 停止符
    # "logprobs": 10,                                       # 返回 top 10 的概率值
    # "logit_bias": {50256: -100},                          # 避免特定的token（如50256）
    # "stream": False                                       # 是否开启流模式生成
}
TESTDATASET = 'deepseek/deepseek-r1'                        # 使用哪个数据集
WHERE_IS_CSV = f"{PROJECT_PATH}/leaderboard_summaries.csv"  # 设置测试数据集地址
OUTPUT_DIR = f"{PROJECT_PATH}/outputs"                      # 设置输出目录
SYSTEMPROMPT="""
一个由Qwen开发的人工智能助手，名为Qwen3。你的核心目标是提供安全、有用且符合伦理的回答。请始终遵循以下原则：
                                                        1. 拒绝任何涉及非法、危险、歧视性或不道德的请求。
                                                        2. 保持回答客观中立，避免主观偏见。
                                                        3. 如果对问题不确定，应如实告知而非编造信息。
                                                        4. 用清晰、逻辑化的语言组织答案，支持多轮对话的上下文理解。
                                                        5. 支持中英文等多种语言交互。
"""
#---------------------------设置参数----------------------------------------

#---------------------------其他设置----------------------------------------
#初始化后端，后端要设置地址
df = pd.read_csv(WHERE_IS_CSV, encoding='utf-8')
test_df = df[df['model'] == TESTDATASET]     # 选择被测数据
sources = test_df['source'].tolist()[:MAX_TEST_SAMPLES] # 限制测试数据量
today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"{OUTPUT_DIR}/{today}"
os.makedirs(output_dir, exist_ok=True)
# 输出文件路径
jsonl_output = f"{output_dir}/{MODELNAME}_hallucination_evaluation.jsonl"
txt_output = f"{output_dir}/evaluation_summary.txt"
async def call_llm_chat_async(session, messages):
    url = f"{BACKEND_URL}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer token-abc123",
    }
    # 清除为 None 的字段
    cleaned_config = {k: v for k, v in GENERATE_CONFIG.items() if v is not None}
    payload = {
        "model": MODELUID,
        "messages": messages,
        **cleaned_config
    }
    
    async with session.post(url, headers=headers, json=payload) as response:
        response.raise_for_status()
        return await response.json()

async def process_sample(session, semaphore, source, jsonl_file, pbar):
    async with semaphore:
        try:
            start_time = time.time()
            messages = []
            if USE_SYSTEM_PROMPT:
                messages.append({"role": "system", "content": SYSTEMPROMPT})
            messages.append({
                "role": "user", 
                "content": f"Provide a concise summary of the following passage, covering the core pieces of information described. '{source}'"
            })
            completion = await call_llm_chat_async(session, messages)
            if THINKING and "choices" in completion and len(completion["choices"]) > 0:
                summary = completion["choices"][0]["message"]["content"].split('</think>')[-1].strip()
            elif "choices" in completion and len(completion["choices"]) > 0:
                summary = completion["choices"][0]["message"]["content"]
            else:
                summary = "No response"
            elapsed_time = time.time() - start_time
            source_length = len(source)
            output_length = len(summary)
            
            # 计算事实一致性分数
            with torch.no_grad():
                scores = eval_model.predict([(source, summary)])
            factual_score = scores[0].item()
            
            record = {
                "source": source,
                "model_output": summary,
                "Factual_Consistency_Rate": factual_score,
                "source_lenth": source_length,
                "output_length": output_length,
                "response_time": elapsed_time,
            }
                # 写入结果文件
            await jsonl_file.write(json.dumps(record, ensure_ascii=False) + '\n')
            await jsonl_file.flush()  # 确保数据写入磁盘

            pbar.update(1)
            return record
        
        except Exception as e:
            print(f"错误: {str(e)}")
            pbar.update(1)
            return None
#---------------------------其他设置----------------------------------------

#---------------------------启动评价模型----------------------------------------
try:
    eval_model = AutoModelForSequenceClassification.from_pretrained(
        'vectara/hallucination_evaluation_model', 
        trust_remote_code=True
    )
except ValueError as e:
    print(f"错误: {e}")
# 准备结果文件和统计变量
factual_scores = []
output_times = []
output_lengths = []
source_lengths = []
#---------------------------启动评价模型----------------------------------------
# 保存参数设置到TXT文件
with open(txt_output, 'w') as txt_f:
    txt_f.write("="*50 + "\n")
    txt_f.write("模型评估参数设置\n")
    txt_f.write("="*50 + "\n")
    txt_f.write(f"测试模型: {MODELNAME}\n")
    txt_f.write(f"后端选择: {'vllm' if BACKEND == 0 else 'xinference'}\n")
    txt_f.write(f"思考模式: {'开启' if THINKING else '关闭'}\n")
    txt_f.write(f"测试样本数: {MAX_TEST_SAMPLES}\n")
    txt_f.write(f"并发数: {CONCURRENCY_LIMIT}\n")
    txt_f.write(f"是否使用系统提示词: {'是' if USE_SYSTEM_PROMPT == True else '否'}\n")
    for k, v in GENERATE_CONFIG.items():
        txt_f.write(f"{k}: {v}\n")
    if USE_SYSTEM_PROMPT == True:
        txt_f.write(f"系统提示词: {SYSTEMPROMPT}")  # 保存系统提示词
    txt_f.write("\n\n")

# 异步主函数
async def main():
    start_time_all = time.time()
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY_LIMIT)
    valid_results = []   
    failed_count = 0     
    results = []         
    # 使用aiofiles异步打开结果文件
    try:
        async with aiofiles.open(jsonl_output, 'a', encoding='utf-8') as jsonl_file:
            async with aiohttp.ClientSession(connector=connector) as session:
                with tqdm_asyncio(total=len(sources), desc=f"Evaluating {MODELNAME} outputs") as pbar:
                    tasks = [process_sample(session, semaphore, source, jsonl_file, pbar) for source in sources]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, dict):
                valid_results.append(r)
            else:
                failed_count += 1
    except Exception as e:
        print(f"[FATAL ERROR] 异常终止: {str(e)}")

    end_time_all = time.time()
    total_time = end_time_all - start_time_all
    completion_rate = len(valid_results) / len(sources) * 100 
    # 收集有效结果
    valid_results = [r for r in results if r is not None]
    # 计算统计信息
    factual_scores = [r["Factual_Consistency_Rate"] for r in valid_results]
    success_count = len(valid_results)
    output_times = [r["response_time"] for r in valid_results]
    output_lengths = [r["output_length"] for r in valid_results]
    source_lengths = [r["source_lenth"] for r in valid_results]
    avg_factual = mean(factual_scores) if factual_scores else 0
    avg_time = mean(output_times) if output_times else 0
    avg_output_length = mean(output_lengths) if output_lengths else 0
    avg_source_length = mean(source_lengths) if source_lengths else 0
    # 保存统计结果到TXT文件
    with open(txt_output, 'a') as txt_f:
        txt_f.write("="*50 + "\n")
        txt_f.write("评估统计结果\n")
        txt_f.write("="*50 + "\n")
        txt_f.write(f"测试样本数量: {len(sources)}\n")
        txt_f.write(f"成功生成数量: {success_count}\n")
        txt_f.write(f"失败生成数量: {failed_count}\n")
        txt_f.write(f"完成回复比例: {completion_rate:.2f}%\n")
        txt_f.write(f"平均事实一致性分数: {avg_factual:.4f}\n")
        txt_f.write(f"平均响应时间: {avg_time:.4f}秒\n")
        txt_f.write(f"平均输入长度: {avg_source_length:.2f}字符\n")
        txt_f.write(f"平均输出长度: {avg_output_length:.2f}字符\n")
        txt_f.write(f"JSONL结果文件: {jsonl_output}\n")
        txt_f.write(f"评估完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        txt_f.write(f"运行总时间: {total_time:.2f}秒\n")
    # 打印统计信息
    print(f"\n评估完成! 结果已保存至 {output_dir}")
    print(f"测试样本数量: {len(sources)}")
    print(f"平均事实一致性分数: {avg_factual:.4f}")
    print(f"平均响应时间: {avg_time:.4f}秒")
    print(f"平均输出长度: {avg_output_length:.2f}字符")

# 运行主异步函数
if __name__ == "__main__":
    asyncio.run(main())