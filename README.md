# machine_learning
* 国科大课程笔记
	* 模式识别与机器学习：https://blog.csdn.net/weixin_40485502/category_9651112.html和上面部分，代码实现ai算法中有部分
	* 大数据分析：https://blog.csdn.net/weixin_40485502/category_9664900.html
	* 高级人工智能：https://blog.csdn.net/weixin_40485502/category_9647785.html
	* 宗老师2018视频学习nlp：https://blog.csdn.net/weixin_40485502/category_9666345.html
* 自我学习
	* ai算法
	* ner
		* HMM[我的HMM博客](https://blog.csdn.net/weixin_40485502/article/details/103900184）
			* 维特比
			* 前向算法
			* 后向算法
			* EM(鲍姆-韦尔奇算法)：80%
				* with scaling的前向，后向
			* 频次统计：87%
		* CRF[我的CRF笔记](https://blog.csdn.net/weixin_40485502/article/details/104094857)
			* 三种表达方式
				* 一般参数
				* 简化参数形式
				* 矩阵形式
				* 一般参数->简化参数形式t_s2f()
				* 简化参数形式->矩阵形式f2M()
			* 三种计算p(y|x)
				* 一般参数P_y_x_condition(self,y)
				* 简化参数形式P_y_x_condition_with_f(self, y)
				* 矩阵形式P_y_x_condition_with_M(y)
				* 用前向后向算法计算$P(y_i|x)$:p_y_x_condition_alpha_beta(self,alpha, beta)
				* 联合概率 （前向后向方法计算$p(y_{i-1}，y_i|x)$
					*  p_y12_x_condition_alpha_beta(self,alpha, beta)
			* 前向后向算法
				* alpha(self)
				* beta(self)
			* 期望
				* $E_{p(y|x)}(f_k)$:E_fk_py_x(self,k, alpha, beta)$E_{p(y|x)}(f_k)$
			* 维特比
				* Viterbi_M(self)
			* 学习
				* 计算delta
				* 一个样本（一句话），根据特征函数可以计算出一组self.f
				* 目前缺点速度及其慢
			* CRF--对应于统计学习书上例11.1
			* CRF-M---是对ner进行计算。
		* LSTM（https://blog.csdn.net/weixin_40485502/article/details/104162822）
			* LSTMbypytorch--使用LSTM实现序列标注（pytorch）
			* myLSTM---自实现LSTM前向及BPTT
			* LSTMbyPytorchforNer---pytorchLSTM,带batch
				* lossfunction带batch输入怎么办？？？
			* LSTMbyPytorchforNer_singlebatch--一次输入一个句子
			* myLSTMforNer---自实现LSTM(用于ner）