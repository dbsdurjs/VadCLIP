CUDA_VISIBLE_DEVICES=0 python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(server_exp18).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_ucf_caption(server_exp18).pth' --kernel=1 --using-caption
CUDA_VISIBLE_DEVICES=0 python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(server_exp18).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_ucf_caption(server_exp18).pth' --kernel=1 --using-caption --save-test-result

CUDA_VISIBLE_DEVICES=0 python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(server_exp18).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_ucf_caption(server_exp18).pth' --kernel=2 --using-caption
CUDA_VISIBLE_DEVICES=0 python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(server_exp18).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_ucf_caption(server_exp18).pth' --kernel=2 --using-caption --save-test-result

CUDA_VISIBLE_DEVICES=0 python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(server_exp18).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_ucf_caption(server_exp18).pth' --kernel=3 --using-caption
CUDA_VISIBLE_DEVICES=0 python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(server_exp18).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_ucf_caption(server_exp18).pth' --kernel=3 --using-caption --save-test-result

CUDA_VISIBLE_DEVICES=0 python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(server_exp18).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_ucf_caption(server_exp18).pth' --kernel=4 --using-caption
CUDA_VISIBLE_DEVICES=0 python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(server_exp18).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_ucf_caption(server_exp18).pth' --kernel=4 --using-caption --save-test-result

CUDA_VISIBLE_DEVICES=0 python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(server_exp18).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_ucf_caption(server_exp18).pth' --kernel=5 --using-caption
CUDA_VISIBLE_DEVICES=0 python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(server_exp18).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_ucf_caption(server_exp18).pth' --kernel=5 --using-caption --save-test-result

CUDA_VISIBLE_DEVICES=0 python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(server_exp18).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_ucf_caption(server_exp18).pth' --kernel=6 --using-caption
CUDA_VISIBLE_DEVICES=0 python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(server_exp18).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_ucf_caption(server_exp18).pth' --kernel=6 --using-caption --save-test-result

CUDA_VISIBLE_DEVICES=0 python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(server_exp18).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_ucf_caption(server_exp18).pth' --kernel=7 --using-caption
CUDA_VISIBLE_DEVICES=0 python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(server_exp18).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_ucf_caption(server_exp18).pth' --kernel=7 --using-caption --save-test-result

# CUDA_VISIBLE_DEVICES=0 python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(server_exp18).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_ucf_caption(server_exp18).pth' --kernel=8 --using-caption
# CUDA_VISIBLE_DEVICES=0 python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(server_exp18).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_ucf_caption(server_exp18).pth' --kernel=8 --using-caption --save-test-result


# server 컴 실험

# CUDA_VISIBLE_DEVICES=0 python src/ucf_train.py --model-path='../vadclip_pth/model/model_ucf_caption(server_exp18).pth' --checkpoint-path='../vadclip_pth/model/checkpoint_ucf_caption(server_exp18).pth' --using-caption
# CUDA_VISIBLE_DEVICES=0 python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_caption(server_exp18).pth' --checkpoint-path '../vadclip_pth/model/checkpoint_ucf_caption(server_exp18).pth' --using-caption --save-test-result
