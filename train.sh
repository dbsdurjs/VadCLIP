# idea4(all_caption) branch
# caption 사용 : clip feat = np.concat(clip feat, cap feat)
# concat 방식

# python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf.pth' --checkpoint-path='../vadclip_pth/model/checkpoint.pth'
# python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf.pth' --checkpoint-path '../vadclip_pth/model/checkpoint.pth'

# change_lr은 proj2 부분에 lr만 바꿔서 진행
python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(concat, proj2, change_lr).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_caption(concat, proj2, change_lr).pth' --using-caption
python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(concat, proj2, change_lr).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_caption(concat, proj2, change_lr).pth' --using-caption



