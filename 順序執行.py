import subprocess

print("正在執行第一個程式...")
subprocess.run(["python", "驗證碼的模型訓練2合1.py"], check=True)
print("第一個程式執行完成")

print("正在執行第二個程式...")
subprocess.run(["python", "驗證碼的模型訓練(去干擾.py"], check=True)
print("第二個程式執行完成")

print("正在執行第三個程式...")
subprocess.run(["python", "驗證碼的模型訓練(原圖.py"], check=True)
print("第三個程式執行完成")

print("正在執行第4個程式...")
subprocess.run(["python", "驗證碼的模型訓練2合1.py"], check=True)
print("第4個程式執行完成")