import logging
import os
import shutil
import time

import soundfile
import torch
import torchaudio

import hubert_model
import infer_tool
import utils
from models import SynthesizerTrn
from preprocess_wave import FeatureInput
from wav_temp import merge

logging.getLogger('numba').setLevel(logging.WARNING)


class ModelWrapper(object):
    def __init__(self,
                 model_dir):
        self.model_dir = model_dir
        self.hubert_soft = hubert_model.hubert_soft(os.path.join(model_dir, 'hubert.pt'))
        self.config_path = "./configs/nyarumul.json"
        self.model_path = os.path.join(model_dir, "G.pth")

        self.hps_ms = utils.get_hparams_from_file(self.config_path)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_g_ms = SynthesizerTrn(
            178,
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            n_speakers=self.hps_ms.data.n_speakers,
            **self.hps_ms.model)
        _ = utils.load_checkpoint(self.model_path, self.net_g_ms, None)
        _ = self.net_g_ms.eval().to(self.dev)
        self.featureInput = FeatureInput(self.hps_ms.data.sampling_rate, self.hps_ms.data.hop_length)

        # @markdown **单声道，22050hz，wav格式**

        # @markdown 角色id——猫雷模型：0号为猫雷，1号为？？？

        # @markdown 角色id
        self.speaker_id = "0"  # @param {type:"string"}

    def predict(self, sample_file_name, f_pitch_change):
        print(f_pitch_change)
        # @markdown 人声文件名（不带.wav）
        clean_name = os.path.splitext(os.path.basename(sample_file_name))[0]  # @param {type:"string"}
        # @markdown 伴奏文件名（可以不放伴奏）（不带.wav）
        bgm_name = ""  # @param {type:"string"}
        # @markdown 每次处理的长度，建议30s以内，大了炸显存
        cut_time = "30"  # @param {type:"string"}
        # @markdown 可为正负（升降n个半音）
        # vc_transform = "0"  # @param {type:"string"}
        vc_transform = int(float(f_pitch_change))

        out_audio_name = clean_name
        # 可填写音源文件列表，音源文件格式为单声道22050采样率wav，放置于raw文件夹下
        clean_names = [clean_name]
        # bgm、trans分别对应歌曲列表，若能找到相应文件、则自动合并伴奏，若找不到bgm，则输出干声（不使用bgm合成多首歌时，可只随意填写一个不存在的bgm名）
        bgm_names = [bgm_name]
        # 合成多少歌曲时，若半音数量不足、自动补齐相同数量（按第一首歌的半音）
        trans = [int(vc_transform)]  # 加减半音数（可为正负）s
        # 每首歌同时输出的speaker_id
        id_list = [int(self.speaker_id)]

        # 每次合成长度，建议30s内，太高了爆掉显存(gtx1066一次15s以内）
        cut_time = int(cut_time)

        # 自动补齐
        infer_tool.fill_a_to_b(bgm_names, clean_names)
        infer_tool.fill_a_to_b(trans, clean_names)
        for clean_name, bgm_name, tran in zip(clean_names, bgm_names, trans):
            origin_sample_rate = infer_tool.resample_to_22050(f'./raw/{clean_name}.wav')
            for speaker_id in id_list:
                speakers = ["猫雷", "？？？"]
                out_audio_name = clean_name
                # 清除缓存文件
                infer_tool.del_file("./wav_temp/input/")
                infer_tool.del_file("./wav_temp/output/")

                raw_audio_path = f"./raw/{clean_name}.wav"
                audio, sample_rate = torchaudio.load(raw_audio_path)

                audio_time = audio.shape[-1] / 22050
                if audio_time > 1.3 * int(cut_time):
                    infer_tool.cut(int(cut_time), raw_audio_path, out_audio_name, "./wav_temp/input")
                else:
                    shutil.copy(f"./raw/{clean_name}.wav", f"./wav_temp/input/{out_audio_name}-0.wav")
                file_list = os.listdir("./wav_temp/input")

                count = 0
                for file_name in file_list:
                    source_path = "./wav_temp/input/" + file_name
                    audio, sample_rate = torchaudio.load(source_path)
                    input_size = audio.shape[-1]

                    sid = torch.LongTensor([int(speaker_id)]).to(self.dev)
                    soft = self.get_units(source_path).squeeze(0).cpu().numpy()
                    pitch = self.transcribe(source_path, soft.shape[0], tran)
                    pitch = torch.LongTensor(pitch).unsqueeze(0).to(self.dev)
                    stn_tst = torch.FloatTensor(soft)

                    with torch.no_grad():
                        x_tst = stn_tst.unsqueeze(0).to(self.dev)
                        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.dev)
                        audio = self.net_g_ms.infer(x_tst,
                                                    x_tst_lengths,
                                                    pitch,
                                                    sid=sid,
                                                    noise_scale=.667,
                                                    noise_scale_w=0.8,
                                                    length_scale=1)[0][
                            0, 0].data.float()

                    result_file_sample_rate = int(audio.shape[0] / input_size * 22050)
                    # if origin_sample_rate != result_file_sample_rate:
                    #     audio = torchaudio.transforms.Resample(orig_freq=result_file_sample_rate, new_freq=origin_sample_rate)(audio)[0]
                        # audio = torchaudio.functional.resample(audio, result_file_sample_rate, origin_sample_rate)
                    audio = audio.cpu().numpy()
                    soundfile.write("./wav_temp/output/" + file_name, audio, origin_sample_rate)
                    count += 1
                    print("%s success: %.2f%%" % (file_name, 100 * count / len(file_list)))
                out_path = merge.run(out_audio_name, bgm_name, out_audio_name)
                return out_path

    def get_units(self, path):
        source, sr = torchaudio.load(path)
        source = torchaudio.functional.resample(source, sr, 16000)
        source = source.unsqueeze(0).to(self.dev)
        with torch.inference_mode():
            units = self.hubert_soft.units(source)
            return units

    def transcribe(self, path, length, transform):
        feature_pit = self.featureInput.compute_f0(path)
        feature_pit = feature_pit * 2 ** (transform / 12)
        feature_pit = infer_tool.resize2d_f0(feature_pit, length)
        coarse_pit = self.featureInput.coarse_f0(feature_pit)
        return coarse_pit


if __name__ == '__main__':
    model_dir = "/home/natas/pool1/ai/models/sovits/"
    start_time = time.time()
    model = ModelWrapper(model_dir=model_dir)
    load_time = time.time()
    print("load_time:{}".format(load_time - start_time))
    test_file = "Mixdown(2)"
    out_path = model.predict(test_file)
    predict_time = time.time()
    print("predict_time:{}".format(predict_time - load_time))
    print(out_path)
