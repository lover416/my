import pandas as pd
 
 
def handle_data(data):
    """
    将数据处理成字典，用于保存样本数据中的类别数据存储情况
    :param data: dataframe 数据源
    :return:样本数据中的类别数据字典，分类结果字典
    """
    # 初始化类别数据字典
    cate_dict = {}
    # 数据集表头列表（各个条件及分类结果）
    header_list = data.columns.tolist()
    # 条件列表
    factor_list = header_list[:-1]
    # 分类结果所在位置
    k = len(header_list) - 1
 
    # result_dict 为分类的结果类型字典
    result_dict = dict(data.iloc[:, k].value_counts())
    # 或使用如下语句：
    # result_dict = dict(data.iloc[:, -1].value_counts())
    result_dict_key = result_dict.keys()
 
    # 将每个分类结果写入 cate_dict
    # 循环各个分类结果
    for result_key in result_dict_key:
        # 如果类别数据字典不存在该分类结果，默认设置空字典
        if result_key not in cate_dict:
            # dict.setdefault(key, default=None)  键不存在于字典中，将会添加键并将值设为默认值
            cate_dict.setdefault(result_key, {})
        # 在该分类结果下，循环各个条件（因素）
        for factor in factor_list:
            # 如果该分类结果字典不存在该条件（因素），默认设置空字典
            if factor not in cate_dict[result_key]:
                cate_dict[result_key].setdefault(factor, {})
            # 获取该条件的分类列表
            factor_key_list = data[factor].value_counts().index.tolist()
            # 循环获取该条件的各个分类数量
            for key in factor_key_list:
                # 获取该分类结果下，该因素中某个分类的数量
                number = data[(data[header_list[k]] == result_key) & (data[factor] == key)].shape[0]
                if key not in cate_dict[result_key][factor]:
                    cate_dict[result_key][factor].setdefault(key, number)
    return cate_dict, result_dict
 
 
def calculate(cate_dict, result_dict, new_data):
    """
    对每个待预测得结果进行贝叶斯公式得计算，并得出预测类别与概率
    :param cate_dict: 样本数据中的类别数据字典
    :param result_dict: 分类结果字典
    :param new_data: 待预测的数据集
    :return: 预测结果列表
    """
    # 获取数据集的各个条件（因素）列表
    factor_list = new_data.columns.tolist()
    # 初始化预测结果列表
    result_list = []
    # 分类结果列表
    result_key_list = cate_dict.keys()
 
    # 循环预测新数据
    for i in range(len(new_data)):
        new_result_dict = {}
        # 循环计算各个分类指标的概率
        for result_key in result_key_list:
            # 该分类结果在所有分类结果中的占比
            all_ratio = result_dict[result_key] / sum(list(result_dict.values()))
 
            # 循环获取该分类结果下，该因素中各个 分类 在 该分类结果 中的占比
            for factor in factor_list:
                ratio = cate_dict[result_key][factor][new_data.iloc[i, factor_list.index(factor)]] / result_dict[result_key]
                # 总占比 乘以 该因素下的各个分类占比
                all_ratio *= ratio
            new_result_dict.setdefault(result_key, all_ratio)
 
        print(new_result_dict)
        # 获取占比最大的分类结果
        max_result_key = max(new_result_dict, key=new_result_dict.get)
        # 获取占比最大的分类结果的占比
        max_value = new_result_dict[max_result_key]
 
        result_list.append([max_result_key, max_value])
    return result_list
 
 
if __name__ == '__main__':
    file_path = "./朴素贝叶斯数据集.xlsx"
    data = pd.read_excel(file_path)
    print("数据源\n", data)
    # 待预测数据
    new_data = pd.DataFrame({"买车": "是", "孩子 ": "有"}, index=[0])
    cate_dict, result_dict = handle_data(data)
    print(cate_dict)
    print(result_dict)
    result = calculate(cate_dict, result_dict, new_data)
    print(result)
