CUDA_VISIBLE_DEVICES=1 python src/xd_train.py --model-path='../vadclip_pth/model/model_xd_caption(server_exp13).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_xd_caption(server_exp13).pth' --using-caption
CUDA_VISIBLE_DEVICES=1 python src/xd_test.py --model-path '../vadclip_pth/model/model_xd_caption(server_exp13).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_xd_caption(server_exp13).pth' --using-caption --save-test-result

# server 컴 실험