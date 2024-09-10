import sqlite3
import os
import base64

# project dependencies
from deepface import DeepFace
from deepface.commons.logger import Logger

'''
人脸与数据库的比对不一定需要逐个计算，尤其是在数据库中有大量人脸数据时，逐个计算会非常耗时。可以考虑以下几种优化方法：

使用索引结构：

KD树或Ball树：这些数据结构可以加速高维空间中的最近邻搜索，适合用于人脸特征的快速比对。
LSH（局部敏感哈希）：通过将相似的特征映射到同一个桶中，可以快速找到潜在的匹配。
特征降维：

使用PCA（主成分分析）或t-SNE等降维技术，减少特征的维度，从而加快计算速度。
批量处理：

如果需要比对多个图像，可以将它们的特征提取和比对过程批量化，减少计算时间。
并行计算：

利用多线程或GPU加速，进行并行计算，特别是在处理大量数据时，可以显著提高效率。
使用深度学习模型：

一些深度学习模型（如FaceNet）可以在特征空间中进行更高效的比对，利用模型的特性来减少计算量。
缓存机制：

对于频繁查询的人脸特征，可以使用缓存机制，避免重复计算。
通过这些方法，可以显著提高人脸与数据库比对的效率，减少逐个计算的需求。
'''

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

    



