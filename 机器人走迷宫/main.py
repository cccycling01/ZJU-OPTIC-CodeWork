# 导入相关包 
import os
import random
import numpy as np
from Maze import Maze
from Runner import Runner
from QRobot import QRobot
from ReplayDataSet import ReplayDataSet
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot # PyTorch版本
#from keras_py.MinDQNRobot import MinDQNRobot as KerasRobot # Keras版本
import matplotlib.pyplot as plt

# 机器人移动方向
move_map = {
    'u': (-1, 0), # up
    'r': (0, +1), # right
    'd': (+1, 0), # down
    'l': (0, -1), # left
}


# 迷宫路径搜索树
class SearchTree(object):


    def __init__(self, loc=(), action='', parent=None):
        """
        初始化搜索树节点对象
        :param loc: 新节点的机器人所处位置
        :param action: 新节点的对应的移动方向
        :param parent: 新节点的父辈节点
        """
        self.loc = loc  # 当前节点位置
        self.to_this_action = action  # 到达当前节点的动作
        self.parent = parent  # 当前节点的父节点
        self.children = []  # 当前节点的子节点

    def add_child(self, child):
        """
        添加子节点
        :param child:待添加的子节点
        """
        self.children.append(child)

    def is_leaf(self):
        """
        判断当前节点是否是叶子节点
        """
        return len(self.children) == 0
    
def back_propagation(node):
    """
    回溯并记录节点路径
    :param node: 待回溯节点
    :return: 回溯路径
    """
    path = []
    while node.parent is not None:
        path.insert(0, node.to_this_action)
        node = node.parent
    return path


def neighbor_not_visit(maze, is_visit_m, node):
    """
    当前节点是否有相邻节点未访问过   
    :param node:待检查节点
    :return:未访问过的邻居节点集合children
    """
    children=[]
    can_move = maze.can_move_actions(node.loc)
    for a in can_move:
        new_loc = tuple(node.loc[i] + move_map[a][i] for i in range(2))
        if not is_visit_m[new_loc]:
            child = SearchTree(loc=new_loc, action=a, parent=node)
            children.append(child)
    return children

def my_search(maze):
    """
    任选深度优先搜索算法、最佳优先搜索（A*)算法实现其中一种
    :param maze: 迷宫对象
    :return :到达目标点的路径 如：["u","u","r",...]
    """
   # -----------------请实现你的算法代码--------------------------------------    
    start = maze.sense_robot()
    root = SearchTree(loc=start)
    queue = [root]  # 节点队列，用于层次遍历
    h, w, _ = maze.maze_data.shape
    is_visit_m = np.zeros((h, w), dtype=np.int)  # 标记迷宫的各个位置是否被访问过
    path = []  # 记录路径
    current_node = queue[0]
    #start_time = datetime.datetime.now()
    i=0
    while True:
        #end_time = datetime.datetime.now()
        #time_cost = end_time - start_time
        #print(str(time_cost).split('.')[0])
        current_node = queue[0]
        queue.pop(0)#从栈顶弹出一个节点作为当前节点
        if current_node.loc == maze.destination:  # 到达目标点
            path = back_propagation(current_node)
            break
        else:
            is_visit_m[current_node.loc] = 1  # 标记当前节点位置已访问
            children = neighbor_not_visit(maze, is_visit_m, current_node)
            for child in children:
                queue.append(child) 
    # -----------------------------------------------------------------------
    return path

# 导入相关包 
import random
import numpy as np
import torch
from QRobot import QRobot
from ReplayDataSet import ReplayDataSet
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot # PyTorch版本
import matplotlib.pyplot as plt
from Maze import Maze
import time

class Robot(TorchRobot):
    def __init__(self, maze):
        """
        初始化 Robot 类
        :param maze:迷宫对象
        """
        super(Robot, self).__init__(maze)
        
        # destination 设置为和迷宫大小相关，为了在足够大的迷宫中，机器人也能「看得到」终点。
        maze.set_reward(reward={
            "hit_wall": 10.,
            "destination": -maze.maze_size ** 2 * 4.,
            "default": 1.,
        })
        self.maze = maze
        self.epsilon = 0
        """开启金手指，获取全图视野"""
        self.memory.build_full_view(maze=maze)
        
        # 初始化后即开始训练
        self.loss_list = self.train()
        

    def train(self):
        loss_list = []
        batch_size = len(self.memory)
        
        start = time.time()
        # 训练，直到能走出这个迷宫
        while True:
            loss = self._learn(batch=batch_size)
            loss_list.append(loss)
            self.reset()
            for _ in range(self.maze.maze_size ** 2 - 1):
                a, r = self.test_update()
                if r == self.maze.reward["destination"]:
                    print('Training time: {:.2f} s'.format(time.time() - start))
                    return loss_list

    def train_update(self):
        """
        以训练状态选择动作并更新Deep Q network的相关参数
        :return :action, reward 如："u", -1
        """
        #action, reward = "u", -1.0

        # -----------------请实现你的算法代码--------------------------------------
        state = self.sense_state()
        action = self._choose_action(state)
        reward = self.maze.move_robot(action)
        # -----------------------------------------------------------------------
        """---update the step and epsilon---"""
        # self.epsilon = max(0.01, self.epsilon * 0.995)
        return action, reward

    def test_update(self):
        """
        以测试状态选择动作并更新Deep Q network的相关参数
        :return : action, reward 如："u", -1
        """
        #action, reward = "u", -1.0
        state = np.array(self.sense_state(), dtype=np.int16)
        state = torch.from_numpy(state).float().to(self.device)

        # -----------------请实现你的算法代码--------------------------------------
        self.eval_model.eval()
        with torch.no_grad():
            q_value = self.eval_model(state).cpu().data.numpy()
        action = self.valid_action[np.argmin(q_value).item()]
        reward = self.maze.move_robot(action)

        # -----------------------------------------------------------------------

        return action, reward
