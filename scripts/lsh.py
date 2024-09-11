# from lshash import LSHash

# # 创建一个LSH对象，指定维度和桶的数量
# lsh = LSHash(hash_size=128, input_dim=128, num_hashtables=1)

# # 假设有一个高维的float数组
# float_array = [0.1, 0.2, 0.3, ..., 0.128]

# # 将float数组添加到LSH中
# lsh.index(float_array)

# # 查询与给定float数组相似的近邻
# query_array = [0.2, 0.3, 0.4, ..., 0.128]
# nearest_neighbors = lsh.query(query_array)

# # 打印查询结果
# for neighbor in nearest_neighbors:
#     print(neighbor)