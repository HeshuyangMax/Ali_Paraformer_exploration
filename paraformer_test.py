import wandb

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 计算两个字符串的最短编辑距离
# ref: 标注答案字符串
# hyp: 模型推理结果字符串
# return: 整型编辑距离
def levenshtein_distance(ref: str, hyp: str) -> int:
    assert type(ref) == str
    assert type(hyp) == str

    row = len(ref) + 1
    column = len(hyp) + 1

    cache = [ [0] * column for i in range(row) ]

    for i in range(row):
        for j in range(column):
            if i == 0 and j == 0:
                cache[i][j] = 0
            elif i == 0 and j != 0:
                cache[i][j] = j
            elif j == 0 and i != 0:
                cache[i][j] = i
            else:
                if ref[i - 1] == hyp[j - 1]:
                    cache[i][j] = cache[i - 1][j - 1]
                else:
                    replace = cache[i - 1][j - 1] + 1
                    insert = cache[i][j - 1] + 1
                    remove = cache[i - 1][j] + 1
 
                    cache[i][j] = min(replace, insert, remove)

    return cache[row - 1][column - 1]      

# 计算字错率
# ref: 标注答案字符串
# hyp: 模型推理结果字符串
# return: 字错率浮点型
def cer(ref: str, hyp: str) -> float:
    assert type(ref) == str
    assert type(hyp) == str

    dis = levenshtein_distance(ref, hyp)
    cerOutput = dis / len(ref)
    return cerOutput

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    
    # track hyperparameters and run metadata
    config={}
)

audio_web = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav'
audio_in_henan = 'audio_data/henan/T0045G0001S0001/T0045G0001S0001.wav'

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')

rec_result = inference_pipeline(audio_in=audio_in_henan)
hyp = rec_result["text"].strip()
print("hypothesis value: ", hyp)

# 使用 'r' 参数来打开文件进行读取
with open('audio_data/henan/T0045G0001S0001/T0045G0001S0001.txt', 'r') as file:  
    # 使用 read() 方法读取文件内容
    data = file.read()  
ref = data.strip()
# 现在，data 变量包含了文件的所有内容
print("reference value: ", ref)

loss = cer(ref=ref, hyp=hyp)

# log metrics to wandb
wandb.log({"loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()