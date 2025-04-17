python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(server_exp7).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_caption(server_exp7).pth' --using-caption
python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(server_exp7).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_caption(server_exp7).pth' --using-caption --save-test-result

python src/xd_train.py --model-path='../vadclip_pth/model/model_xd_caption(server_exp7).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_caption(server_exp7).pth' --using-caption
python src/xd_test.py --model-path '../vadclip_pth/model/model_xd_caption(server_exp7).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_caption(server_exp7).pth' --using-caption --save-test-result

# idea66(30) learnable prompt(caption)  + position embedding 적용, 메모리 너무 많이 차지 및 배치 32로 줄임, 성능 좋지 않아서 사용 안 할 예정