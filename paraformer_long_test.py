from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    vad_model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    #punc_model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
    punc_model='damo/punc_ct-transformer_cn-en-common-vocab471067-large',
)

rec_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.wav', 
                                batch_size_token=5000, batch_size_token_threshold_s=40, max_single_segment_time=6000)
print(rec_result)
