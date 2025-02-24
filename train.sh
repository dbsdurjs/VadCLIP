# idea6(all_caption) branch
# caption 사용 : clip feat = np.concatenate(clip feat, cap feat)
# 부족한 부분의 padding을 마지막 부분의 값들로 pad

# python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf.pth' --checkpoint-path='../vadclip_pth/model/checkpoint.pth'
# python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf.pth' --checkpoint-path '../vadclip_pth/model/checkpoint.pth'

# python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(padding).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_caption(padding).pth' --using-caption
python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(padding).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_caption(padding).pth' --using-caption

# /home/yeogeon/YG_main/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/Extracted_Frames/Arson/Arson011_x264/Arson011_x264_visualization.mp4