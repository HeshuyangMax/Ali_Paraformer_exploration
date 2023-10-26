from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online',
    model_revision='v1.0.7',
    update_model=False,
    mode='paraformer_streaming'
    )
import soundfile
speech, sample_rate = soundfile.read("example/asr_example.wav")

chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention
param_dict = {"cache": dict(), "is_final": False, "chunk_size": chunk_size,
              "encoder_chunk_look_back": encoder_chunk_look_back, "decoder_chunk_look_back": decoder_chunk_look_back}
chunk_stride = chunk_size[1] * 960 # 600ms、480ms
# first chunk, 600ms
speech_chunk = speech[0:chunk_stride]
rec_result = inference_pipeline(audio_in=speech_chunk, param_dict=param_dict)
print(rec_result)
# next chunk, 600ms
speech_chunk = speech[chunk_stride:chunk_stride+chunk_stride]
rec_result = inference_pipeline(audio_in=speech_chunk, param_dict=param_dict)
print(rec_result)
