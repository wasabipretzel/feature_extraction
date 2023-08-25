import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import torchvision
import torch.nn as nn
import decord
from decord import VideoReader
from decord import cpu, gpu
import tqdm 

# device=  'cpu'
# model = torchvision.models.video.r3d_18(pretrained=True)


# # 폴더 지정
# save_폴더 지정 
# 폴더 내 파일 돌아가면서 파일명으로 video 읽고 extract해서 다시 save_폴더에 파일명으로 저장
def convert_avi_to_mp4(avi_file_path, output_name):
    os.popen(f"ffmpeg -i {avi_file_path} -strict -2 {output_name}.mp4")
    return True

def convert_all_mp4():
    base_path = '/data/max2action/videos'
    inner_vid = os.listdir(base_path)

    avi_list = []
    folder_list = []
    for single_path in inner_vid:
        if '.avi' in single_path:
            avi_list.append(single_path)
        else:
            folder_list.append(single_path)
    for single_vid in avi_list:
        convert_avi_to_mp4(os.path.join(base_path,single_vid), single_vid.split('.')[0])
# import os


def convert_avi_to_mp4(avi_file_path, output_name):
    os.popen(f"ffmpeg -i {avi_file_path} -strict -2 {output_name}.mp4")
    return True
# breakpoint()
    # convert_avi_to_mp4('/data/max2action/videos/CATER_new_003107.avi', 'CATER_new_003107')



def extract_single_vid(model, vr, sampling_rate, output_dim):
    #vr -> T, W, H, C
    vr = vr[::32,] 
    # vr[::32]
    T, W, H, C = vr.shape
    vr = vr.reshape(C, T, W, H).unsqueeze(0).to(dtype=torch.float32) # B(1) C, T, W, H -> input of r3d_18
    # vr = torchvision.transforms.functional.to_tensor(vr)
    # breakpoint()
    device=  'cpu'
    # img = torch.randn(1, 3, 192, 112, 112).to(device)
    # result = model(img)

    feature_extract = nn.Sequential(
        model.stem,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        model.avgpool
    )
    result = torch.tensor([])
    output_dim = 512
    for f_num in range(vr.shape[2]):
        target_frame = vr[:,:,f_num,].unsqueeze(2) #[B(1), C, T(1), W, H]
        result = torch.cat([result, feature_extract(target_frame).reshape(target_frame.shape[0], output_dim)], dim=0)
    return result

def main(model, sampling_rate, output_dim, target_path):
    base_path = '/data/max2action/mp4_videos'
    inner_vid = os.listdir(base_path)

    mp4_list = []
    folder_list = []
    for single_path in inner_vid:
        if '.mp4' in single_path:
            mp4_list.append(single_path)
        else:
            folder_list.append(single_path)
    
    for single_vid in tqdm.tqdm(mp4_list):
        #single mp4 video
        vr = torch.tensor(VideoReader(os.path.join(base_path, single_vid), ctx=cpu(0))[:].asnumpy()) #T, W, H, C
        feature = extract_single_vid(model, vr, sampling_rate, output_dim) #[T, output_dim]
        # save by torch
        torch.save(feature, f"{target_path}/{single_vid.split('.')[0]}.pt")




if __name__ == '__main__':
    sampling_rate = 32
    output_dim=512
    target_path = '/data/max2action/r3d/features'
    model = torchvision.models.video.r3d_18(pretrained=True)

    # convert_all_mp4()
    main(model, sampling_rate, output_dim, target_path)

