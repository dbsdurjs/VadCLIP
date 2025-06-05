CUDA_VISIBLE_DEVICES=0 python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(server_exp14).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_caption(server_exp14).pth' --using-caption
CUDA_VISIBLE_DEVICES=0 python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(server_exp14).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_caption(server_exp14).pth' --using-caption --save-test-result

# server 컴 실험