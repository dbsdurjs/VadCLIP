python src/ucf_train.py --model-path="../vadclip_pth/model/model_ucf_base_train.pth" --checkpoint-path='../vadclip_pth/model/checkpoint_base_train.pth' --seed=254
python src/ucf_test.py --model-path '../vadclip_pth/model/model_ucf_base_train.pth' --checkpoint-path '../vadclip_pth/model/checkpoint_base_train.pth' --seed=254

# python src/ucf_test.py --model-path '../vadclip_pth/model_ucf_pretrained.pth'
