import os

def rename_repeated_prefix_files(dir_path):
    """
    dir_path 내부의 파일들 중에서
    예: "Arson019_x264_Arson019_x264_frame_00000.jpg"
         -> "Arson019_x264_frame_00000.jpg"
    형태로 이름을 일괄 변경.
    """
    for filename in os.listdir(dir_path):
        # 확장자가 .jpg 가 아닌 경우는 무시 (원하는 경우만 처리)
        if not filename.lower().endswith(".jpg"):
            continue

        # 예: "Arson019_x264_Arson019_x264_frame_00000.jpg"
        # 언더바로 split
        # splitted = ["Arson019","x264","Arson019","x264","frame","00000.jpg"]
        splitted = filename.split("_")

        # 최소 길이 확인 (예: [0,1,2,3,4,5] => 6개 이상이어야 함)
        if len(splitted) < 6:
            continue

        # 앞 2개와 다음 2개가 동일하면 "중복 prefix"라고 판단
        #   splitted[0:2] == splitted[2:4]
        # 예: splitted[0:2] = ["Arson019","x264"]
        #     splitted[2:4] = ["Arson019","x264"]
        if splitted[0:2] == splitted[2:4]:
            # 새 파일명은 앞 2개 + (뒤쪽 2개를 제외한 나머지)
            # 즉 splitted[4:] 부분은 ["frame","00000.jpg"]
            new_name = "_".join(splitted[0:2] + splitted[4:])
            # new_name = "Arson019_x264_frame_00000.jpg"

            old_path = os.path.join(dir_path, filename)
            new_path = os.path.join(dir_path, new_name)

            print(f"[Rename] {filename}  ->  {new_name}")
            os.rename(old_path, new_path)

# --------------------------------------------------
if __name__ == "__main__":
    # 예: "C:/my_folder" 나 "/home/username/..."
    target_dir = '/home/yeogeon/YG_main/diffusion_model/VAD_dataset/UCF-Crimes/UCF_Crimes/Extracted_Frames/Arson/Arson019_x264'
    rename_repeated_prefix_files(target_dir)
