import numpy as np

# 创建包含128维float数组的数据集
data = np.random.rand(100, 128)

# 计算数据集的均值
mean = np.mean(data, axis=0)

# 将数据集减去均值，得到零均值数据
centered_data = data - mean

# 计算协方差矩阵
covariance_matrix = np.cov(centered_data, rowvar=False)

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# 将特征向量按特征值从大到小排序
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# 选择前k个特征向量作为主成分
k = 2
principal_components = sorted_eigenvectors[:, :k]

# 将数据投影到主成分上
projected_data = np.dot(centered_data, principal_components)

# 打印投影后的数据
print(projected_data)