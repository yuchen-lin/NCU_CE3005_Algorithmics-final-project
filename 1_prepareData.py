# import之後要用到的librarys
import shutil
import os
from PIL import Image

# 為了方便訓練時讀檔，我將train_image和test_image資料路徑處理成:
# ---------------------
# /train_image
#   /0
#       /1.bmp
#       /2.bmp
#       /...
#   /1
#       /6.bmp
#       /7.bmp
#       /...
#   /2
#   /...
# ---------------------
# 就是把資料拉出來，去掉學號那層folder

# 準備資料路徑
train_path = './train_image'
test_path = './test_image'
# 準備第二層folder中的學號名稱
train_studentID = os.listdir(train_path)
test_studentID = os.listdir(test_path)

# 建立資料夾
for i in range(10):
    os.mkdir(train_path + '/' + str(i))
    os.mkdir(test_path + '/' + str(i))

# 將train_image中的資料去除學號那層folder
count = 1
for s in train_studentID:
    for i in os.listdir(train_path + '/' + s):
        for j in os.listdir(train_path + '/' + s + '/' + i):
            # file_path為原始圖檔的完整路徑
            file_path = train_path + '/' + s + '/' + i + '/' + j
            # 因為要同類圖片放入同資料夾，所以將抓出的圖片重新命名為連續數字
            save_path = train_path + f'/{i}/{count}.png'
            # 將.bmp轉為.png方便訓練
            Image.open(file_path).resize((28,28)).save(save_path)
            count += 1

# 將test_image中的資料去除學號那層folder
count = 1
for s in test_studentID:
    for i in os.listdir(test_path + '/' + s):
        for j in os.listdir(test_path + '/' + s + '/' + i):
            # file_path為原始圖檔的完整路徑
            file_path = test_path + '/' + s + '/' + i + '/' + j
            # 因為要同類圖片放入同資料夾，所以將抓出的圖片重新命名為連續數字
            save_path = test_path + f'/{i}/{count}.png'
            # 將.bmp轉為.png方便訓練
            Image.open(file_path).resize((28,28)).save(save_path)
            count += 1

# 刪除空的學號folder
for i in train_studentID:
    shutil.rmtree(train_path + '/' + i, ignore_errors=True)
for i in test_studentID:
    shutil.rmtree(test_path + '/' + i, ignore_errors=True)