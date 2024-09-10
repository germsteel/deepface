import sqlite3
import os
import base64

# project dependencies
from deepface import DeepFace
from deepface.commons.logger import Logger

# 连接到数据库（如果不存在则创建）
# conn = sqlite3.connect('mydatabase.db')

# # 创建一个游标对象来执行SQL语句
# cursor = conn.cursor()

# # 创建表
# cursor.execute('''
#     CREATE TABLE mytable (
#         id INT PRIMARY KEY NOT NULL,
#         float_array BLOB,
#         image_base64 TEXT,
#         image_name TEXT
#     )
# ''')

# # 提交更改并关闭连接
# conn.commit()
# conn.close()

def generate_imbedding(img_path):
    model_name = "Facenet"
    # 获得嵌入向量
    img_embedding = DeepFace.represent(img_path=img_path, model_name=model_name)[0]["embedding"]
    return img_embedding


def transform_image_to_base64(img_path):
    with open(img_path, "rb") as image:
        f = image.read()
        image_base64 = base64.b64encode(f).decode("utf-8")
    return image_base64

def traversal_image(path):
    # 遍历文件夹
    index = 0
    conn = sqlite3.connect('mydatabase.db')
    cursor = conn.cursor()
    for root, dirs, files in os.walk(path):
        for file in files:
            img_path = os.path.join(root, file)
            img_embedding = generate_imbedding(img_path)
            image_base64 = transform_image_to_base64(img_path)
            cursor.execute('''
                INSERT INTO mytable (id, float_array, image_base64, image_name) VALUES (?, ?, ?, ?)
            ''', (index, img_embedding, image_base64, file))
            # 提交更改并关闭连接
            index += 1

    conn.commit()
    conn.close()

path = os.path.join(os.getcwd(), "deepface\\tests\\dataset")
traversal_image(path)

    



