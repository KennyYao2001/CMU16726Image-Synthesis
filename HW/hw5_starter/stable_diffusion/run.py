import os
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a painting of a virus monster playing guitar", 
                        help="the prompt to render")
    parser.add_argument("--input-img", type=str, required=True, 
                        help="path to the input image")
    parser.add_argument("--seed", type=int, default=10, 
                        help="the seed (for reproducible sampling)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 要测试的不同参数组合
    # strengths = [15.0, 25.0, 35.0]
    # timesteps = [500, 600, 700]
    strengths = [35.0]
    timesteps = [600]
    
    # 确保输出目录存在
    os.makedirs("outputs", exist_ok=True)
    
    # 运行每种组合
    for strength in strengths:
        for timestep in timesteps:
            # 为每个组合创建一个输出文件夹
            output_dir = f"outputs/strength_{strength}_timestep_{timestep}"
            os.makedirs(output_dir, exist_ok=True)
            
            # 构建命令
            cmd = [
                "python", "img2img.py",
                "--prompt", args.prompt,
                "--input-img", args.input_img,
                "--strength", str(strength),
                "--num_timesteps", str(timestep),
                "--seed", str(args.seed)
            ]
            
            print(f"\nRunning with strength={strength}, timestep={timestep}")
            print(" ".join(cmd))
            
            # 修改 img2img.py 中的输出路径
            # 这使用临时文件修改，不会永久更改原始代码
            with open("img2img.py", "r") as f:
                content = f.read()
            
            modified_content = content.replace(
                'Image.fromarray(out_image.astype(np.uint8)).save(f"outputs/{i:05}.png")',
                f'Image.fromarray(out_image.astype(np.uint8)).save(f"{output_dir}/{{i:05}}_std2_{os.path.basename(args.input_img)}.png")'
            )
            
            with open("img2img_temp.py", "w") as f:
                f.write(modified_content)
            
            # 执行修改后的脚本
            process = subprocess.Popen(
                cmd[0:1] + ["img2img_temp.py"] + cmd[2:],
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            
            print(stdout.decode())
            if stderr:
                print("错误信息:", stderr.decode())
    
    # 清理临时文件
    if os.path.exists("img2img_temp.py"):
        os.remove("img2img_temp.py")
    
    print("\n所有组合已完成!")

if __name__ == "__main__":
    main()