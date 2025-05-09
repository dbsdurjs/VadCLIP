CUDA_VISIBLE_DEVICES=0 python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(server_exp10).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_caption(server_exp10).pth' --using-caption
CUDA_VISIBLE_DEVICES=0 python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(server_exp10).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_caption(server_exp10).pth' --using-caption --save-test-result

# server_exp10 기반 idea66(35) 기반 수정, 로컬 컴퓨터에서 실행