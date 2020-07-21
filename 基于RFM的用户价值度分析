import time
import numpy as np
import pandas as pd
import pymysql




# 读取原始数据
dtypes = {'ORDERDATE': object, 'ORDERID': object, 'AMOUNTINFO': np.float}
raw_data = pd.read_csv('sales.csv', dtype=dtypes, index_col='USERID')



# 数据概览、缺失值审查
# print(raw_data.describe())
"""
对DataFrame来说，describe()默认情况下，只返回数字字段。
describe(include='all')返回数据的所有列。
"""



na_cols = raw_data.isnull().any(axis=0)     # 查看每一列是否有缺失值
# print(na_cols)
na_lines = raw_data.isnull().any(axis=1)    # 查看每一行是否有缺失值
# print('总的NA行数：{}'.format(na_lines.sum()))
# print(raw_data[na_lines])   # 查看具有缺失值的行信息



# 异常值处理
sales_data = raw_data.dropna()
sales_data = sales_data[sales_data['AMOUNTINFO'] > 1]   # 丢弃金额≤1
# 日期格式转换
sales_data['ORDERDATE'] = pd.to_datetime(sales_data['ORDERDATE'], format='%Y-%m-%d')
"""
format参数以原始数据字符串格式来写，只有格式对应才能实现解析
"""
# print(sales_data.dtypes)



# 计算R、F、M
recency_value = sales_data['ORDERDATE'].groupby(sales_data.index).max()     # 计算最近一次订单时间
frequency_value = sales_data['ORDERDATE'].groupby(sales_data.index).count() # 计算频率
monetary_value = sales_data['AMOUNTINFO'].groupby(sales_data.index).sum()   # 计算总金额
"""
groupby() 函数可以进行数据的分组以及分组后的组内运算。
print(df["评分"].groupby([df["地区"],df["类型"]]).mean())
该条语句的功能：输出数据中不同地区不同类型的评分的平均值。
"""



# 计算R、F、M得分
deadline_date = pd.datetime(2017, 1, 1)     # 指定时间点，计算其他时间与该时间的距离
r_interval = (deadline_date - recency_value).dt.days            # 计算 R 间隔
r_score = pd.cut(r_interval, 5, labels=[5, 4, 3, 2, 1])         # 计算 R 得分
f_score = pd.cut(frequency_value, 5, labels=[1, 2, 3, 4, 5])    # 计算 F 得分
m_score = pd.cut(monetary_value, 5, labels=[1, 2, 3, 4, 5])     # 计算 M 得分
# print(r_score.head())



# 合并数据框
rfm_list = [r_score, f_score, m_score]
rfm_cols = ['r_score', 'f_score', 'm_score']    # 设置R、F、M三个维度列名
rfm_pd = pd.DataFrame(np.array(rfm_list).T, dtype=np.int32, columns=rfm_cols, index=frequency_value.index)    # 建立R、F、M数据框
"""
np.array().transpose()等价于np.array().T都表示数组的转置
dtype=np.int32 等价于 dtype='int32'
"""
# print(rfm_pd.head(4))



# 计算RFM总得分
# 方法一：加权得分
rfm_pd['rfm_wscore'] = rfm_pd['r_score']*0.6 + rfm_pd['f_score']*0.3 + rfm_pd['m_score']*0.1
# 方法二：RFM组合
rfm_pd_tmp = rfm_pd.copy()
rfm_pd_tmp['r_score'] = rfm_pd_tmp['r_score'].astype('str')
rfm_pd_tmp['f_score'] = rfm_pd_tmp['f_score'].astype('str')
rfm_pd_tmp['m_score'] = rfm_pd_tmp['m_score'].astype('str')
rfm_pd['rfm_comb'] = rfm_pd_tmp['r_score'].str.cat(rfm_pd_tmp['f_score']).str.cat(rfm_pd_tmp['m_score'])
# print(rfm_pd.head())



# 连接mysql数据库
# 设置要写库的数据库连接信息
table_name = 'sales_rfm_score'      # 要写库的表名
# 数据库基本信息
config = {'host': '127.0.0.1',
          'user': 'root',
          'password': 'root',
          'port': 3306,
          'database': 'python_data',
          'charset': 'gb2312'}
con = pymysql.connect(**config)     # 建立mysql连接
cursor = con.cursor()               # 获得游标
# 查找数据库是否存在目标表，如果没有则新建
cursor.execute('show tables')
table_object = cursor.fetchall()    # 通过fetchall方法获取数据
table_list = []                     # 创建库列表
for t in table_object:              # 循环读出多有库
    table_list.append(t[0])         # 每个库追加到列表
if not table_name in table_list:    # 如果目标表没有创建
    cursor.execute("""
    CREATE TABLE %s (
    userid      VARCHAR (20),
    r_score     int(2),
    f_score     int(2),
    m_score     int(2),
    rfm_wscore  DECIMAL (10, 2),
    rfm_comb    VARCHAR (10),
    insert_date VARCHAR (20)
    )ENGINE=InnoDB DEFAULT CHARSET=gb2312
    """ % table_name)               # 创建新表
# 将数据写入数据库
user_id = rfm_pd.index  # 索引列
rfm_wscore = rfm_pd['rfm_wscore']   # RFM加权得分列
rfm_comb = rfm_pd['rfm_comb']       # RFM组合得分列
timestamp = time.strftime('%Y-%m-%d', time.localtime(time.time()))      # 写库日期
print('Begin to insert data into table {0}...'.format(table_name))     # 输出开始写库的提示信息
for i in range(rfm_pd.shape[0]):    # 设置循环次数并依次循环
    insert_sql = "INSERT INTO `%s` VALUES ('%s',%s,%s,%s,%s,'%s','%s')" % \
                 (table_name, user_id[i], r_score.iloc[i], f_score.iloc[i], m_score.iloc[i], rfm_wscore.iloc[i],
                  rfm_comb.iloc[i], timestamp)  # 写库SQL依据
    cursor.execute(insert_sql)                  # 执行SQL语句，execute函数里面要用双引号
    con.commit()    # 提交命令
cursor.close()      # 关闭游标
con.close()         # 关闭数据库连接
print('Finish inserting, total records is: %d' % (i + 1))  # 打印写库结果



