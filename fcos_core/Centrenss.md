2024.3.18
修改目标：将Centerness加入到loss中。
第一步，接口引入。
fcos_core/modeling/rpn/fcos/fcos.py 186-205-213行 获取
fcos_core/engine/trainer.py 在62 65得到  送入69 行
fcos_core/modeling/rpn/fcos/graph_matching_head.py 在212行左右进行接受修改
fcos_core/modeling/rpn/fcos/loss.py 采样区域使用完毕

第二部 loss 修改

2024.3.22
loss修改回1

2024年3月26日

发现loss没有修改意义，其原因在于我设置的loss没有梯度，而是单纯的数值