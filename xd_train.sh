
CUDA_VISIBLE_DEVICES=1 python src/xd_train.py --model-path='../vadclip_pth/model/graduate_paper_base_xd.pth' --checkpoint-path='../vadclip_pth/model/graduate_paper_base_xd.pth' --using-caption
CUDA_VISIBLE_DEVICES=1 python src/xd_test.py --model-path '../vadclip_pth/model/graduate_paper_base_xd.pth' --checkpoint-path '../vadclip_pth/model/graduate_paper_base_xd.pth' --save-test-result
