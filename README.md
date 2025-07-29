# vectare-hallucination-evaluation-model 项目说明
本项目使用 Hugging Face 的 Hallucination Evaluation Model（幻觉评估模型）对生成文本进行事实一致性评估。以下是使用说明。

📦 创建 Conda 环境

conda create -n test python=3.10
conda activate test
🛠️ 安装依赖
✅ 国内用户（使用清华镜像）

pip install pandas torch transformers tqdm aiohttp aiofiles -i https://pypi.tuna.tsinghua.edu.cn/simple
✅ 国外用户（常规 PyPI 源）

pip install pandas torch transformers tqdm aiohttp aiofiles

📦 安装评估模型（国内用户需要参考下列教程，国外用户自动忽略即可）

✅ 指定镜像源（国内用户推荐）：
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoModelForSequenceClassification
pairs = [ # Test data, List[Tuple[str, str]]
    ("The capital of France is Berlin.", "The capital of France is Paris."), # factual but hallucinated
    ('I am in California', 'I am in United States.'), # Consistent
    ('I am in United States', 'I am in California.'), # Hallucinated
    ("A person on a horse jumps over a broken down airplane.", "A person is outdoors, on a horse."),
    ("A boy is jumping on skateboard in the middle of a red bridge.", "The boy skates down the sidewalk on a red bridge"),
    ("A man with blond-hair, and a brown shirt drinking out of a public water fountain.", "A blond man wearing a brown shirt is reading a book."),
    ("Mark Wahlberg was a fan of Manny.", "Manny was a fan of Mark Wahlberg.")
]
🛠️ 自定义模型依赖修改
在以下路径中找到文件并进行修改：
.cache/huggingface/hub/models--vectara--hallucination_evaluation_model/snapshots/c9ccea35f0e37e02422eb49b798e09ed82be5a520/configuration_hhem_v2.py
将变量 foundation 设置为以下路径：
.cache/huggingface/hub/models--google--flan-t5-base/snapshots/7bcac572ce56db69c1ea7c8af255c5d7c9672fc2

🧪 修改配置参数（run.py）
请进入项目目录后，点击打开 run.py 并修改以下关键参数：

ACKEND_URL = http://0.0.0.0:8001;    # 后端服务器地址
MODELUID = deepseek                  # 模型UID，必须修改
THINKING = True                      # 是否开启思考模式
📁 项目输出说明
运行完成后，会在项目目录下生成一个新的 outputs 文件夹，文件夹名称以日期命名（如 2025-07-28_07-21-24），夹内包含以下内容：

输出JSONL文件：qwen3-14B_hallucination_evaluation.jsonl
结果TXT文件：包含评估结果的文本摘要
✅ 示例运行结果输出

==================================================
模型评估参数设置
==================================================
测试模型: qwen3-14B
后端选择: xinference
思考模式: 关闭
测试样本数: 5
是否使用系统提示词: 是
并发数：10
max_tokens: 1024
top_k: 50
temperature: 0.7
frequency_penalty: 0.5
系统提示词: 
一个由Qwen开发的人工智能助手，名为Qwen3。你的核心目标是提供安全、有用且符合伦理的回答。请始终遵循以下原则：
                                                        1. 拒绝任何涉及非法、危险、歧视性或不道德的请求。
                                                        2. 保持回答客观中立，避免主观偏见。
                                                        3. 如果对问题不确定，应如实告知而非编造信息。
                                                        4. 用清晰、逻辑化的语言组织答案，支持多轮对话的上下文理解。
                                                        5. 支持中英文等多种语言交互。
==================================================
评估统计结果
==================================================
测试样本数量: 200
成功生成数量: 200
失败生成数量: 0
完成回复比例: 100.00%
平均事实一致性分数: 0.8772
平均响应时间: 8.8906秒
平均输入长度: 1218.97字符
平均输出长度: 457.67字符
JSONL结果文件: path/outputs/2025-07-28_07-21-24/qwen3-14B_hallucination_evaluation.jsonl
评估完成时间: 2025-07-28 07:23:00
运行总时间: 95.32秒

📌 注意事项
模型下载可能较慢，建议使用镜像源或网络稳定时进行操作；
对于国内无法直接访问 Hugging Face 的用户，建议使用 hf-mirror.com 进行镜像访问，如上文操作步骤；

✅ 项目运行确认
创建 conda 环境并激活；
安装所有依赖；
修改模型配置和 run.py 参数；
运行项目；
检查 outputs 文件夹中是否生成评估结果。
如需更多帮助，请提供具体问题或错误日志内容。


